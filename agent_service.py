import os
import io
import json
import uuid
import base64
import asyncio
import wave
import audioop
import hashlib
import time
import re
import struct
from typing import Dict, Optional, List, Tuple
from collections import deque

from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.responses import Response, PlainTextResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient

# PyTorch (for GPU detection)
import torch

# RAG stack
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

import httpx

# Deepgram SDK
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions,
)

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from datetime import datetime as dt


# ================================
# DATABASE MODELS (Agent Management)
# ================================
Base = declarative_base()

class Agent(Base):
    """Agent configuration (like ElevenLabs Agent)"""
    __tablename__ = "agents"
    
    agent_id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    first_message = Column(Text, nullable=True)
    
    # Voice settings
    voice_provider = Column(String(50), default="deepgram")  # deepgram, elevenlabs, etc.
    voice_id = Column(String(100), default="aura-2-thalia-en")
    
    # Model settings
    model_provider = Column(String(50), default="ollama")
    model_name = Column(String(100), default="mixtral:8x7b")
    
    # Behavior settings
    interrupt_enabled = Column(Boolean, default=True)
    silence_threshold_sec = Column(Float, default=0.8)
    
    # Metadata
    created_at = Column(DateTime, default=dt.utcnow)
    updated_at = Column(DateTime, default=dt.utcnow, onupdate=dt.utcnow)
    user_id = Column(String(100), nullable=True)  # For multi-tenancy
    is_active = Column(Boolean, default=True)


class Conversation(Base):
    """Store conversation history (like ElevenLabs Conversation)"""
    __tablename__ = "conversations"
    
    conversation_id = Column(String(100), primary_key=True)  # Twilio call_sid
    agent_id = Column(String(100), nullable=False)
    
    # Call details
    phone_number = Column(String(50), nullable=True)
    status = Column(String(50), default="initiated")  # initiated, in-progress, completed, failed
    
    # Transcript
    transcript = Column(Text, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_secs = Column(Integer, default=0)
    
    # Metadata (renamed to avoid SQLAlchemy reserved word)
    dynamic_variables = Column(JSON, nullable=True)  # Lead data
    call_metadata = Column(JSON, nullable=True)  # Call metadata (was 'metadata')
    
    # Results
    ended_reason = Column(String(100), nullable=True)
    recording_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=dt.utcnow)


class WebhookConfig(Base):
    """Webhook configuration for call events"""
    __tablename__ = "webhook_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=True)  # null = global webhook
    
    webhook_url = Column(String(500), nullable=False)
    events = Column(JSON, default=list)  # ["call.started", "call.ended", etc.]
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


class PhoneNumber(Base):
    """Phone numbers linked to agents"""
    __tablename__ = "phone_numbers"
    
    id = Column(String(100), primary_key=True)
    phone_number = Column(String(50), nullable=False, unique=True)
    agent_id = Column(String(100), nullable=True)  # Linked agent
    provider = Column(String(50), default="twilio")
    label = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


class KnowledgeBase(Base):
    """Knowledge base documents per agent"""
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False)
    document_id = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    kb_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.utcnow)


class AgentTool(Base):
    """Custom tools per agent"""
    __tablename__ = "agent_tools"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False)
    tool_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    webhook_url = Column(String(500), nullable=True)  # External webhook for tool
    parameters = Column(JSON, default=dict)  # Tool parameters schema
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


# Database connection
DATABASE_URL = os.getenv("AGENT_DATABASE_URL", "sqlite:///./agents.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================================
# PYDANTIC MODELS (API Requests)
# ================================

class CallRequest(BaseModel):
    to_number: str


class AgentCreate(BaseModel):
    name: str
    system_prompt: str
    first_message: Optional[str] = None
    voice_provider: str = "deepgram"
    voice_id: str = "aura-2-thalia-en"
    model_provider: str = "ollama"
    model_name: str = "mixtral:8x7b"
    interrupt_enabled: bool = True
    silence_threshold_sec: float = 0.8


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    voice_provider: Optional[str] = None
    voice_id: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    interrupt_enabled: Optional[bool] = None
    silence_threshold_sec: Optional[float] = None
    is_active: Optional[bool] = None


class OutboundCallRequest(BaseModel):
    """ElevenLabs-compatible outbound call request"""
    agent_id: str
    agent_phone_number_id: Optional[str] = None  # For compatibility
    to_number: str
    first_message: Optional[str] = None  # Override agent's default

    conversation_initiation_client_data: Optional[Dict] = Field(default_factory=dict)
    enable_recording: bool = False  # Enable call recording


class WebhookCreate(BaseModel):
    """Create webhook for agent events"""
    webhook_url: str = Field(..., description="URL to send webhook events to")
    events: List[str] = Field(
        default_factory=lambda: ["call.initiated", "call.started", "call.ended"],
        description="List of events to subscribe to"
    )
    agent_id: Optional[str] = Field(None, description="Agent ID (null for global webhook)")


class WebhookResponse(BaseModel):
    """Webhook response"""
    success: bool
    webhook_id: int
    webhook_url: str
    events: List[str]
    agent_id: Optional[str] = None


load_dotenv()
# ----------------------------
# Environment and configuration
# ----------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_VOICE = os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en")
DATA_FILE = os.getenv("DATA_FILE", "./data/data.json")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mixtral:8x7b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "384"))
TOP_K = int(os.getenv("TOP_K", "3"))

# 🎯 SMART INTERRUPT SETTINGS - ALL CONFIGURABLE
INTERRUPT_ENABLED = os.getenv("INTERRUPT_ENABLED", "true").lower() == "true"
INTERRUPT_MIN_ENERGY = int(os.getenv("INTERRUPT_MIN_ENERGY", "1000"))
INTERRUPT_DEBOUNCE_MS = int(os.getenv("INTERRUPT_DEBOUNCE_MS", "1000"))
INTERRUPT_BASELINE_FACTOR = float(
    os.getenv("INTERRUPT_BASELINE_FACTOR", "3.5"))
INTERRUPT_MIN_SPEECH_MS = int(os.getenv("INTERRUPT_MIN_SPEECH_MS", "300"))
INTERRUPT_REQUIRE_TEXT = os.getenv(
    "INTERRUPT_REQUIRE_TEXT", "false").lower() == "true"

# âœ… SILENCE DETECTION (matches Deepgram utterance_end_ms)
SILENCE_THRESHOLD_SEC = float(os.getenv("SILENCE_THRESHOLD_SEC", "0.8"))
UTTERANCE_END_MS = int(SILENCE_THRESHOLD_SEC * 1000)

REQUIRE_ENV = [TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
               TWILIO_PHONE_NUMBER, PUBLIC_URL, DEEPGRAM_API_KEY]
if not all(REQUIRE_ENV):
    raise RuntimeError(
        "Missing required env: TWILIO_*, PUBLIC_URL, DEEPGRAM_API_KEY")

# JWT Secret for signed URLs
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")

# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="xi-api-key", auto_error=False)
API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []

# Webhook Events
WEBHOOK_EVENTS = [
    "call.initiated",
    "call.started", 
    "call.ended",
    "call.failed",
    "transcript.partial",
    "transcript.final",
    "agent.response",
    "tool.called",
    "user.interrupted"
]


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key - returns None if no API_KEYS configured (dev mode)"""
    # If no API keys configured, allow all requests (dev mode)
    if not API_KEYS or API_KEYS == ['']:
        return None
    
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
LOG_FILE = os.getenv("LOG_FILE", "server.log")

_logger = logging.getLogger("new")
_logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
_logger.addHandler(_ch)

try:
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
    _fh.setFormatter(_fmt)
    _logger.addHandler(_fh)
except Exception:
    pass

# ----------------------------
# 🚀 GPU DETECTION & OPTIMIZATION
# ----------------------------


def detect_gpu():
    """Detect and configure GPU"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda

        _logger.info("=" * 60)
        _logger.info("🚀 GPU DETECTED!")
        _logger.info(f"   Device: {gpu_name}")
        _logger.info(f"   Count: {gpu_count}")
        _logger.info(f"   Memory: {gpu_memory:.2f} GB")
        _logger.info(f"   CUDA: {cuda_version}")
        _logger.info("=" * 60)

        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _logger.info("âœ… TF32 enabled (Ampere+ GPU)")

        torch.cuda.empty_cache()

    elif torch.backends.mps.is_available():
        device = 'mps'
        _logger.info("=" * 60)
        _logger.info("🚀 Apple Silicon GPU detected")
        _logger.info("=" * 60)
    else:
        device = 'cpu'
        _logger.warning("=" * 60)
        _logger.warning("âš ï¸  NO GPU DETECTED - Using CPU")
        _logger.warning("=" * 60)

    return device


DEVICE = detect_gpu()

_logger.info("🚀 Config: PUBLIC_URL=%s DEVICE=%s", PUBLIC_URL, DEVICE)
_logger.info("🎯 Interrupt: ENABLED=%s MIN_SPEECH=%dms MIN_ENERGY=%d BASELINE_FACTOR=%.1f",
             INTERRUPT_ENABLED, INTERRUPT_MIN_SPEECH_MS, INTERRUPT_MIN_ENERGY, INTERRUPT_BASELINE_FACTOR)
_logger.info("â±ï¸  Silence Threshold: %.1fs (utterance_end=%dms)",
             SILENCE_THRESHOLD_SEC, UTTERANCE_END_MS)


def public_ws_host() -> str:
    host = PUBLIC_URL.replace(
        "https://", "").replace("http://", "").rstrip("/")
    return host


# ----------------------------
# Clients
# ----------------------------
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

deepgram_config = DeepgramClientOptions(
    options={"keepalive": "true", "timeout": "60"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, config=deepgram_config)

# ----------------------------
# 🚀 GPU-ACCELERATED RAG
# ----------------------------
_logger.info(f"📦 Loading SentenceTransformer on {DEVICE}...")
start_time = time.time()

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
embedder.eval()

if DEVICE == 'cuda':
    try:
        embedder.half()
        _logger.info("âœ… FP16 precision enabled")
    except Exception as e:
        _logger.warning(f"âš ï¸  Could not enable FP16: {e}")

load_time = time.time() - start_time
_logger.info(f"âœ… Model loaded in {load_time:.2f}s")

_logger.info("🔥 Warming up GPU...")
with torch.no_grad():
    _ = embedder.encode(
        ["warmup sentence for GPU initialization"],
        device=DEVICE,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=1
    )
_logger.info("âœ… GPU warmed up")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("docs")

response_cache = {}


# ðŸ› ï¸ LLM-CONTROLLED TOOL SYSTEM
# ----------------------------


async def end_call_tool(call_sid: str, reason: str = "user_goodbye") -> dict:
    """End the active call"""
    _logger.info(f"ðŸ”š END_CALL: call_sid={call_sid}, reason={reason}")

    try:
        await asyncio.sleep(1.5)
        call = twilio_client.calls(call_sid).update(status="completed")
        await manager.disconnect(call_sid)

        return {
            "success": True,
            "message": f"Call ended: {reason}",
            "call_sid": call_sid
        }
    except Exception as e:
        _logger.error(f"âŒ Failed to end call: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def transfer_call_tool(call_sid: str, department: str = "sales") -> dict:
    """Transfer call to human agent - executes AFTER message is spoken"""
    _logger.info(f"ðŸ”€ TRANSFER_CALL: call_sid={call_sid}, dept={department}")

    DEPARTMENT_NUMBERS = {
        "sales": os.getenv("SALES_PHONE_NUMBER", "+918107061392"),
        "support": os.getenv("SUPPORT_PHONE_NUMBER", "+918107061392"),
        "technical": os.getenv("TECH_PHONE_NUMBER", "+918107061392"),
    }

    try:
        transfer_number = DEPARTMENT_NUMBERS.get(
            department, DEPARTMENT_NUMBERS["sales"])

        conn = manager.get(call_sid)
        if not conn:
            return {"success": False, "error": "Connection not found"}

        _logger.info("â³ Waiting for transfer message to be spoken...")
        await asyncio.sleep(3.0)

        conn.interrupt_requested = True

        while not conn.tts_queue.empty():
            try:
                conn.tts_queue.get_nowait()
                conn.tts_queue.task_done()
            except:
                break

        try:
            await conn.ws.send_json({
                "event": "clear",
                "streamSid": conn.stream_sid
            })
        except:
            pass

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>{transfer_number}</Dial>
</Response>"""

        twilio_client.calls(call_sid).update(twiml=twiml)

        _logger.info(
            f"âœ… Transfer completed to {department} ({transfer_number})")

        return {
            "success": True,
            "transfer_to": transfer_number,
            "department": department,
            "message": f"Transferred to {department}"
        }

    except Exception as e:
        _logger.error(f"âŒ Transfer failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def detect_confirmation_response(text: str) -> Optional[str]:
    """
    Detect if user is confirming or rejecting a pending action.
    Returns: "yes", "no", or None
    """
    text_lower = text.lower().strip()

    yes_patterns = [
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please",
        "go ahead", "do it", "that's fine", "sounds good",
        "yes please", "yeah please", "sure thing", "absolutely",
        "correct", "right", "affirmative", "proceed", "transfer me",
        "let's do it", "fine", "alright", "all right"
    ]

    no_patterns = [
        "no", "nope", "nah", "not yet", "not now", "maybe later",
        "don't", "wait", "hold on", "cancel", "never mind",
        "not right now", "i'll think about it", "let me think",
        "not really", "not interested"
    ]

    for pattern in yes_patterns:
        if pattern == text_lower or pattern in text_lower:
            if "not " not in text_lower and "no " not in text_lower[:3]:
                return "yes"

    for pattern in no_patterns:
        if pattern == text_lower or pattern in text_lower:
            return "no"

    return None


def parse_llm_response(text: str) -> Tuple[str, Optional[dict]]:
    """
    Parse LLM response for tool calls.
    Format: [TOOL:tool_name:param1:param2] for immediate execution
            [CONFIRM_TOOL:tool_name:param1] for confirmation requests

    Returns: (clean_text, tool_data)
    """

    tool_pattern = r'\[TOOL:([^\]]+)\]'
    confirm_pattern = r'\[CONFIRM_TOOL:([^\]]+)\]'

    tool_data = None

    confirm_matches = re.findall(confirm_pattern, text)
    if confirm_matches:
        tool_parts = confirm_matches[0].split(':')
        tool_name = tool_parts[0].strip()

        if tool_name == "transfer":
            department = tool_parts[1].strip() if len(
                tool_parts) > 1 else "sales"

            valid_departments = ["sales", "support", "technical"]
            if department not in valid_departments:
                _logger.warning(
                    f"âŒ Invalid department in CONFIRM_TOOL: {department} - ignoring tool call")
            else:
                tool_data = {
                    "tool": "transfer_call",
                    "params": {"department": department},
                    "requires_confirmation": True
                }
    else:
        tool_matches = re.findall(tool_pattern, text)
        if tool_matches:
            tool_parts = tool_matches[0].split(':')
            tool_name = tool_parts[0].strip()

            if tool_name == "end_call":
                tool_data = {
                    "tool": "end_call",
                    "params": {"reason": "user_requested"},
                    "requires_confirmation": False
                }
            elif tool_name == "transfer":
                department = tool_parts[1].strip() if len(
                    tool_parts) > 1 else "sales"

                valid_departments = ["sales", "support", "technical"]
                if department not in valid_departments:
                    _logger.warning(
                        f"âŒ Invalid department in TOOL: {department} - ignoring tool call")
                else:
                    tool_data = {
                        "tool": "transfer_call",
                        "params": {"department": department},
                        "requires_confirmation": False
                    }

    clean_text = re.sub(tool_pattern, '', text)
    clean_text = re.sub(confirm_pattern, '', clean_text).strip()

    return clean_text, tool_data


async def execute_detected_tool(call_sid: str, tool_data: dict) -> dict:
    """Execute a tool that was detected from LLM response"""
    tool_name = tool_data["tool"]
    params = tool_data.get("params", {})

    _logger.info(
        f"ðŸ”§ Executing LLM-requested tool: {tool_name} with params: {params}")

    if tool_name == "end_call":
        result = await end_call_tool(call_sid, **params)
    elif tool_name == "transfer_call":
        result = await transfer_call_tool(call_sid, **params)
    else:
        result = {"success": False, "error": f"Unknown tool: {tool_name}"}

    return result

# ----------------------------
# âš¡ GPU-ACCELERATED STREAMING RAG QUERY
# ----------------------------


# ----------------------------
# âš¡ GPU-ACCELERATED STREAMING RAG QUERY (true streaming)
# ----------------------------
# ----------------------------
# âš¡ GPU-ACCELERATED STREAMING RAG QUERY (no tools, true streaming)
# ----------------------------
async def query_rag_streaming(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K,
    call_sid: Optional[str] = None
):
    """✨ ENHANCED: RAG with agent configuration and dynamic variables support"""
    if history is None:
        history = []

    # Get current date in America/New_York timezone
    from datetime import datetime
    import pytz
    ny_tz = pytz.timezone('America/New_York')
    current_date = datetime.now(ny_tz).strftime("%A, %B %d, %Y")
    
    # ✨ Load agent configuration and dynamic variables
    conn = manager.get(call_sid) if call_sid else None
    agent_prompt = None
    dynamic_vars = {}
    model_to_use = OLLAMA_MODEL  # Default from env
    
    model_source = "env_default"
    
    if conn and conn.agent_config:
        agent_prompt = conn.agent_config.get("system_prompt")
        dynamic_vars = conn.dynamic_variables or {}
        _logger.info(f"✅ Using agent prompt with {len(dynamic_vars)} dynamic variables")
        
        # ✨ Use custom model if provided, otherwise agent default, otherwise env default
        if conn.custom_model and conn.custom_model.strip():
            model_to_use = conn.custom_model
            model_source = "api_override"
        elif conn.agent_config.get("model_name"):
            model_to_use = conn.agent_config["model_name"]
            model_source = "agent_config"
    
    _logger.info(f"🤖 Model: {model_to_use} (source: {model_source})")

    loop = asyncio.get_running_loop()

    def _embed_and_query():
        with torch.no_grad():
            query_embedding = embedder.encode(
                [question],
                device=DEVICE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1
            )[0].tolist()

            return collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2
            )

    results = await loop.run_in_executor(None, _embed_and_query)

    raw_docs = results.get("documents", [[]])[0] if results else []
    distances = results.get("distances", [[]])[0] if results else []

    # Simple relevance filtering
    relevant_chunks = []
    for doc, dist in zip(raw_docs, distances):
        if dist <= 1.3:  # Simple threshold
            relevant_chunks.append(doc)

    # Use top 3 most relevant
    context_text = "\n".join(relevant_chunks[:3])

    # _logger.info(f"📚 Found {len(relevant_chunks)} relevant chunks")

    # Build conversation history
    history_text = ""
    if history and len(history) > 0:
        recent_history = history[-6:]  # Keep last 3 exchanges
        history_lines = []
        for h in recent_history:
            history_lines.append(
                f"User: {h['user']}\nAssistant: {h['assistant']}")
        history_text = "\n".join(history_lines)

    # ✨ BUILD PROMPT - Use agent's system_prompt if available, otherwise use default
    if agent_prompt:
        # ✨ AGENT-BASED PROMPT with dynamic variables
        vars_section = ""
        if dynamic_vars:
            vars_lines = []
            for key, value in dynamic_vars.items():
                if value and str(value).strip():
                    vars_lines.append(f"- **{key}**: {value}")
            if vars_lines:
                vars_section = "\n\n## Lead/Customer Information:\n" + "\n".join(vars_lines)
        
        prompt = f"""{agent_prompt}

## Current Date (America/New_York):
Today is {current_date}.{vars_section}

## Knowledge Base Context:
{context_text if context_text.strip() else "No specific context found."}

## Conversation History:
{history_text if history_text else "This is the start of the call."}

## Current Question:
{question}"""
    else:
        prompt = f"""You are MILA, an Outbound AI voice call assistant for Technology Mindz. Technology Mindz provides some of the key services- Salesforce, Artificial Intelligence, Managed IT Services, Cybersecurity, Microsoft Dynamics 365, Staff Augmentation, CRM Consulting, Web Development, Mobile App Development.

## Current Date (America/New_York):
Today is {current_date}. Use this information for scheduling, date validation, and context-aware responses.

## situation:
- you have made an outbound call, you are on a call and talking to a user.
- keep yourself as a human is talking to a human on call.

## STRICT INSTRUCTIONS:
  CRITICAL: Only use the **company knowledge base context** to answer the user's **Current Question**.
- always check user **Current Question** available in **company knowledge base's** data, if is not available than decline to respone about that, but never halluinate.
- Keep responses accurate from the **company knowledge base's** data only.
- Keep responses conversative (that is going to use in tts) with natural filler words.
- Offer to schedule meetings it when relevant.
- For meeting scheduling: ask for date, time, and timezone. Allow only FUTURE dates (not today).
- after getting these details of meeting -> tell user to wait while you are validating the meeting details to be in working hours or working days and output exact this: [TOOL:meeting_call:DATE and time in str format:TIMEZONE in IANA format:user's address based on their timezone] -> if response from tool is "valid : true" then confirm the meeting is scheduled otherwise apologize and ask for rescheduling.
- If you understand that the user wants to finish the conversation or end the call (for example they say "bye", "thank you", "that's all", "nothing else", "talk later", "end the call", or show signs of closing the call), then politely end the conversation and output this exactly: [TOOL:end_call]

## Previous Conversation History on current call:
`{history_text if history_text else "This is the start of the call."}`

## company knowledge base context:
**{context_text}**

## Current Question:
**{question}**

Assistant:"""

    # Rest of your streaming code remains the same...
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    full_response = ""

    def _safe_put(item):
        """Safely put item in queue, handling QueueFull gracefully"""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            # Queue is full - drop to prevent blocking
            # Try to make space by removing oldest item if queue is very full
            if queue.qsize() > 400:
                try:
                    queue.get_nowait()  # Remove one old item
                    queue.put_nowait(item)  # Try again
                except:
                    pass  # If that fails, just drop the item

    def _producer():
        nonlocal full_response
        try:
            for chunk in ollama.generate(
                model=model_to_use,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": 0.2,
                    "num_predict": 1200,
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_ctx": 1024,
                    "num_thread": 8,
                    "repeat_penalty": 1.2,
                    "repeat_last_n": 128,
                    "num_gpu": 99,
                    "stop": ["\nUser:", "\nAssistant:", "User:"],
                }
            ):
                token = chunk.get("response")
                if token:
                    full_response += token
                    loop.call_soon_threadsafe(_safe_put, token)
            loop.call_soon_threadsafe(_safe_put, None)
        except Exception as e:
            loop.call_soon_threadsafe(_safe_put, {"__error__": str(e)})

    loop.run_in_executor(None, _producer)

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict) and "__error__" in item:
                yield "I'm having trouble responding right now. Could you repeat that?"
                return

            # Yield tokens immediately (consumer will decide when to speak)
            yield item

    except Exception as e:
        yield "I'm having trouble answering right now. Could you repeat that?"


def calculate_audio_energy(mulaw_bytes: bytes) -> int:
    """Calculate RMS energy of audio chunk"""
    if not mulaw_bytes or len(mulaw_bytes) < 160:
        return 0
    try:
        pcm = audioop.ulaw2lin(mulaw_bytes, 2)
        return audioop.rms(pcm, 2)
    except Exception:
        return 0

# ----------------------------
# WebSocket Connection Manager
# ----------------------------


class WSConn:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stream_sid: Optional[str] = None
        self.inbound_ulaw_buffer: bytearray = bytearray()
        self.is_responding: bool = False
        self.last_transcript: str = ""
        self.stream_ready: bool = False
        self.speech_detected: bool = False
        self.currently_speaking: bool = False
        self.interrupt_requested: bool = False
        self.conversation_history: List[Dict[str, str]] = []
        
        # ✨ NEW: Agent and call data
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[Dict] = None
        self.dynamic_variables: Optional[Dict] = None
        self.custom_first_message: Optional[str] = None

        self.custom_voice_id: Optional[str] = None
        self.custom_model: Optional[str] = None
        self.conversation_id: Optional[str] = None  # For DB tracking

        # Streaming STT
        self.deepgram_live = None
        self.stt_transcript_buffer: str = ""
        self.stt_is_final: bool = False
        self.last_speech_time: float = 0
        self.silence_start: Optional[float] = None

        # Streaming TTS
        # Limit queue size to prevent memory issues
        self.tts_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.tts_task: Optional[asyncio.Task] = None

        # 🎯 SMART VOICE-BASED INTERRUPT DETECTION
        self.user_speech_detected: bool = False
        self.speech_start_time: Optional[float] = None
        self.speech_energy_buffer: deque = deque(maxlen=50)
        self.last_interrupt_time: float = 0
        self.interrupt_debounce: float = INTERRUPT_DEBOUNCE_MS / 1000.0

        # ✅ Baseline starts at 50% of threshold
        self.baseline_energy: float = INTERRUPT_MIN_ENERGY * 0.5
        self.background_samples: deque = deque(maxlen=50)

        # For smarter interrupt gating
        self.last_interim_text: str = ""
        self.last_interim_time: float = 0.0
        self.last_interim_conf: float = 0.0
        self.last_tts_send_time: float = 0.0

        # ✨ Pending action confirmation
        self.pending_action: Optional[dict] = None

        # 🔧 ADD THIS: Speech validation to prevent false positives
        self.false_speech_check_time: Optional[float] = None

        # VAD validation fields
        self.vad_triggered_time: Optional[float] = None
        self.vad_validation_threshold: float = 0.3
        self.vad_validated: bool = False
        self.vad_timeout: float = 5.0
        self.energy_drop_time: Optional[float] = None
        self.last_valid_speech_energy: float = 0.0

        # 🔧 CRITICAL FIX: Session-level resampler state (prevents clicks between responses)
        self.resampler_state = None
        self.resampler_initialized: bool = False


class ConnectionManager:
    def __init__(self):
        self._conns: Dict[str, WSConn] = {}

    async def connect(self, call_sid: str, ws: WebSocket):
        self._conns[call_sid] = WSConn(ws)

    async def disconnect(self, call_sid: str):
        conn = self._conns.pop(call_sid, None)
        if conn:
            if conn.deepgram_live:
                try:
                    conn.deepgram_live.finish()
                except:
                    pass

            if conn.tts_task and not conn.tts_task.done():
                conn.tts_task.cancel()

            try:
                await conn.ws.close()
            except Exception:
                pass

    def get(self, call_sid: str) -> Optional[WSConn]:
        return self._conns.get(call_sid)

    async def send_media_chunk(self, call_sid: str, stream_sid: str, raw_mulaw_bytes: bytes):
        conn = self.get(call_sid)
        if not conn or not conn.ws or not conn.stream_ready:
            return False

        if conn.interrupt_requested:
            return False

        payload = base64.b64encode(raw_mulaw_bytes).decode("utf-8")

        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": payload
            }
        }

        try:
            await conn.ws.send_json(msg)
            return True
        except Exception as e:
            return False


manager = ConnectionManager()

# ✨ Store for passing call data from API to WebSocket
# Key: call_sid, Value: {agent_id, dynamic_variables, overrides}
pending_call_data: Dict[str, Dict] = {}


async def save_conversation_transcript(call_sid: str, conn: WSConn):
    """
    Save conversation transcript to database
    
    ✨ ELEVENLABS-COMPATIBLE: Always saves transcript, even if empty
    """
    _logger.info(f"💾 save_conversation_transcript called for {call_sid}")
    _logger.info(f"   - conn exists: {bool(conn)}")
    _logger.info(f"   - conversation_history length: {len(conn.conversation_history) if conn else 0}")
    
    if not conn:
        _logger.warning(f"⚠️ No connection found for {call_sid} - cannot save transcript")
        return
    
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            # Build transcript text
            transcript_lines = []
            for entry in conn.conversation_history:
                user_text = entry.get('user', '')
                assistant_text = entry.get('assistant', '')
                transcript_lines.append(f"User: {user_text}")
                transcript_lines.append(f"Assistant: {assistant_text}")
            
            # ✨ Save transcript even if empty (to show call happened)
            conversation.transcript = "\n".join(transcript_lines) if transcript_lines else "[No conversation - call ended early]"
            conversation.status = "completed"
            conversation.ended_at = dt.utcnow()
            
            # Calculate duration
            if conversation.started_at:
                duration = (conversation.ended_at - conversation.started_at).total_seconds()
                conversation.duration_secs = int(duration)
            
            db.commit()
            _logger.info(f"✅ Saved transcript for {call_sid}")
            _logger.info(f"   - Exchanges: {len(conn.conversation_history)}")
            _logger.info(f"   - Duration: {conversation.duration_secs}s")
            _logger.info(f"   - Transcript length: {len(conversation.transcript)} chars")
        else:
            _logger.warning(f"⚠️ Conversation record not found in DB for {call_sid}")
    except Exception as e:
        _logger.error(f"❌ Failed to save transcript: {e}")
        import traceback
        _logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


async def handle_call_end(call_sid: str, reason: str):
    """
    Handle call ending - save data and send webhooks
    
    ✨ ELEVENLABS-COMPATIBLE: Always saves transcript and sends webhooks
    """
    conn = manager.get(call_sid)
    
    # Save transcript
    if conn:
        await save_conversation_transcript(call_sid, conn)
    
    db = SessionLocal()
    try:
        # Update conversation
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            conversation.ended_reason = reason
            conversation.status = "completed"
            if not conversation.ended_at:
                conversation.ended_at = dt.utcnow()
            
            # Calculate duration if not already set
            if conversation.started_at and not conversation.duration_secs:
                duration = (conversation.ended_at - conversation.started_at).total_seconds()
                conversation.duration_secs = int(duration)
            
            db.commit()
            
            # Extract direction from call_metadata
            call_direction = "outbound"
            if conversation.call_metadata and isinstance(conversation.call_metadata, dict):
                call_direction = conversation.call_metadata.get("direction", "outbound")
            
            # ✨ ALWAYS send webhooks (like ElevenLabs)
            webhooks = db.query(WebhookConfig).filter(
                WebhookConfig.is_active == True
            ).all()
            
            for webhook in webhooks:
                should_send = False
                if webhook.agent_id is None:
                    should_send = True  # Global webhook
                elif conversation.agent_id and webhook.agent_id == conversation.agent_id:
                    should_send = True  # Agent-specific webhook
                
                if should_send and ("call.ended" in webhook.events or not webhook.events):
                    await send_webhook(
                        webhook.webhook_url,
                        "call.ended",
                        {
                            "conversation_id": call_sid,
                            "agent_id": conversation.agent_id,
                            "duration_secs": conversation.duration_secs,
                            "ended_reason": reason,
                            "transcript": conversation.transcript,
                            "phone_number": conversation.phone_number,
                            "direction": call_direction,
                            "dynamic_variables": conversation.dynamic_variables,
                            "status": "completed"
                        }
                    )
            
            _logger.info(f"✅ Call ended: {call_sid} - reason: {reason} - duration: {conversation.duration_secs}s")
        else:
            _logger.warning(f"⚠️ Conversation not found for call end: {call_sid}")
    except Exception as e:
        _logger.error(f"❌ Failed to handle call end: {e}")
    finally:
        db.close()

# ----------------------------
# 🎯 SMART VOICE-BASED INTERRUPT DETECTION
# ----------------------------


def update_baseline(conn: WSConn, energy: int):
    """Update background noise baseline with improved adaptivity"""
    if not conn.currently_speaking:
        if energy < max(conn.baseline_energy * 2, 600):
            conn.background_samples.append(energy)
            if len(conn.background_samples) >= 20:
                recent_samples = list(conn.background_samples)[-20:]
                sorted_samples = sorted(recent_samples)
                weighted_median = sorted_samples[len(sorted_samples) // 2]
                conn.baseline_energy = (
                    conn.baseline_energy * 0.7) + (weighted_median * 0.3)


# ----------------------------
# âš¡ Interrupt Handler
# ----------------------------

async def handle_interrupt(call_sid: str):
    """Handle user interruption with complete cleanup"""
    conn = manager.get(call_sid)
    if not conn:
        return

    _logger.info("ðŸ›‘ INTERRUPT - Stopping playback and clearing buffers")

    conn.interrupt_requested = True

    cleared = 0
    while not conn.tts_queue.empty():
        try:
            conn.tts_queue.get_nowait()
            conn.tts_queue.task_done()
            cleared += 1
        except:
            break

    try:
        await conn.ws.send_json({
            "event": "clear",
            "streamSid": conn.stream_sid
        })
    except:
        pass

    old_buffer = conn.stt_transcript_buffer
    conn.stt_transcript_buffer = ""
    conn.stt_is_final = False
    conn.last_transcript = ""

    conn.currently_speaking = False
    conn.is_responding = False
    conn.speech_energy_buffer.clear()
    conn.speech_start_time = None
    conn.user_speech_detected = False
    conn.last_speech_time = 0
    conn.silence_start = None

    conn.last_interim_text = ""
    conn.last_interim_time = 0.0
    conn.last_interim_conf = 0.0

    _logger.info(
        "âœ… Interrupt handled:\n"
        "   Cleared TTS items: %d\n"
        "   Cleared STT buffer: '%s'\n"
        "   Ready for new input",
        cleared, old_buffer[:50] if old_buffer else "(empty)"
    )

# ----------------------------
# âš¡ STREAMING TTS
# ----------------------------


async def stream_tts_worker(call_sid: str):
    """âš¡ OPTIMIZED TTS - Fast first response + smooth playback + no clicks"""
    conn = manager.get(call_sid)
    if not conn:
        return

    # Single resampler for entire session (critical for smoothness)
    # persistent_resampler_state = None

    try:
        while True:
            # âœ… SINGLE SENTENCE: Process one sentence at a time
            text = await conn.tts_queue.get()

            if text is None:
                conn.tts_queue.task_done()
                break

            conn.tts_queue.task_done()

            if not text or not text.strip():
                continue

            if conn.interrupt_requested:
                _logger.info("ðŸ›‘ Skipping batch due to interrupt")
                while not conn.tts_queue.empty():
                    try:
                        conn.tts_queue.get_nowait()
                        conn.tts_queue.task_done()
                    except:
                        break
                conn.currently_speaking = False
                conn.interrupt_requested = False
                # persistent_resampler_state = None
                break

            _logger.info("ðŸŽ¤ TTS sentence (%d chars): '%s...'",
                         len(text), text[:80])

            t_start = time.time()
            conn.currently_speaking = True
            conn.speech_energy_buffer.clear()
            conn.speech_start_time = None
            is_first_chunk = True  # Track first chunk of sentence

            try:
                url = "https://api.deepgram.com/v1/speak"
                headers = {
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {"text": text}
                
                # ✨ Use custom voice if provided, otherwise agent default, otherwise env default
                voice_to_use = DEEPGRAM_VOICE  # Default from env
                voice_source = "env_default"
                
                # 🔍 DEBUG: Log raw values for debugging
                _logger.debug(f"🔍 TTS Voice Debug - conn.custom_voice_id: '{conn.custom_voice_id}'")
                _logger.debug(f"🔍 TTS Voice Debug - conn.agent_config: {conn.agent_config}")
                
                if conn.custom_voice_id and str(conn.custom_voice_id).strip():
                    voice_to_use = conn.custom_voice_id
                    voice_source = "api_override"
                elif conn.agent_config and conn.agent_config.get("voice_id"):
                    voice_to_use = conn.agent_config["voice_id"]
                    voice_source = "agent_config"
                
                # Log voice selection for EVERY sentence (to debug first message issue)
                _logger.info(f"🎤 TTS Voice: {voice_to_use} (source: {voice_source}) for text: '{text[:50]}...'")
                
                params = {
                    "model": voice_to_use,
                    "encoding": "linear16",
                    "sample_rate": "16000"
                }

                interrupted = False
                chunk_count = 0

                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("POST", url, json=payload,
                                             headers=headers, params=params) as response:
                        response.raise_for_status()

                        async for audio_chunk in response.aiter_bytes(chunk_size=3200):
                            if conn.interrupt_requested:
                                _logger.info(
                                    "ðŸ›' TTS interrupted at chunk %d", chunk_count)
                                interrupted = True
                                break

                            if len(audio_chunk) == 0:
                                continue

                            try:
                                # ✅ CRITICAL: Ensure resampler is initialized before first chunk
                                if conn.resampler_state is None:
                                    # Initialize resampler with silence
                                    _, conn.resampler_state = audioop.ratecv(
                                        b'\x00' * 160, 2, 1, 16000, 8000, None
                                    )

                                # ✅ CRITICAL: Reuse same resampler state across all sentences
                                pcm_8k, conn.resampler_state = audioop.ratecv(
                                    audio_chunk, 2, 1, 16000, 8000,
                                    conn.resampler_state
                                )

                                # ✅ FIX: Apply fade-in to first chunk to prevent clicks
                                if is_first_chunk and len(pcm_8k) >= 320:
                                    # Convert to list for manipulation
                                    samples = list(struct.unpack(
                                        f'<{len(pcm_8k)//2}h', pcm_8k))

                                    # Apply fade-in to first 160 samples (10ms at 8kHz)
                                    fade_samples = min(160, len(samples))
                                    for i in range(fade_samples):
                                        fade_factor = (i + 1) / fade_samples
                                        samples[i] = int(
                                            samples[i] * fade_factor)

                                    # Repack
                                    pcm_8k = struct.pack(
                                        f'<{len(samples)}h', *samples)
                                    is_first_chunk = False

                                mulaw = audioop.lin2ulaw(pcm_8k, 2)

                                for i in range(0, len(mulaw), 160):
                                    if conn.interrupt_requested:
                                        interrupted = True
                                        break

                                    chunk_to_send = mulaw[i:i+160]
                                    if len(chunk_to_send) < 160:
                                        chunk_to_send += b'\xff' * \
                                            (160 - len(chunk_to_send))

                                    success = await manager.send_media_chunk(
                                        call_sid, conn.stream_sid, chunk_to_send
                                    )
                                    if not success:
                                        interrupted = True
                                        break

                                    conn.last_tts_send_time = time.time()
                                    chunk_count += 1
                                    await asyncio.sleep(0.018)

                                if interrupted:
                                    break

                            except Exception as e:
                                continue

                t_end = time.time()

                if interrupted:
                    await handle_interrupt(call_sid)
                    # Keep resampler state - don't reset on interrupt
                    while not conn.tts_queue.empty():
                        try:
                            conn.tts_queue.get_nowait()
                            conn.tts_queue.task_done()
                        except:
                            break
                else:
                    _logger.info("âœ… Sentence completed in %.0fms (%d chunks, %.1f chars/sec)",
                                 (t_end - t_start)*1000, chunk_count,
                                 len(text) / (t_end - t_start) if (t_end - t_start) > 0 else 0)

            except Exception as e:
                # ✅ Only reset resampler on serious conversion errors
                if "resampler" in str(e).lower() or "audio" in str(e).lower():
                    conn.resampler_state = None

            # Only clear state when truly done
            if conn.tts_queue.empty():
                conn.currently_speaking = False
                conn.interrupt_requested = False
                conn.speech_energy_buffer.clear()
                conn.speech_start_time = None
                conn.user_speech_detected = False
                # Keep resampler for next turn

    except asyncio.CancelledError:
        pass
    except Exception as e:
        pass
    finally:
        conn.currently_speaking = False
        conn.interrupt_requested = False


async def speak_text_streaming(call_sid: str, text: str):
    """âš¡ Queue text with smart sentence splitting"""
    conn = manager.get(call_sid)
    if not conn or not conn.stream_sid:
        return

    try:
        await conn.ws.send_json({
            "event": "clear",
            "streamSid": conn.stream_sid
        })
    except:
        pass

    conn.currently_speaking = True
    conn.interrupt_requested = False
    conn.speech_energy_buffer.clear()
    conn.user_speech_detected = False

    # âœ… Split into sentences for queue
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    # Queue all sentences (worker will batch them automatically)
    for sentence in sentences:
        if sentence:
            try:
                await asyncio.wait_for(conn.tts_queue.put(sentence), timeout=2.0)
            except asyncio.TimeoutError:
                break
            except Exception as e:
                break

    await conn.tts_queue.join()
    conn.currently_speaking = False

# âš¡ STREAMING STT WITH IMPROVED VAD - Deepgram live + final-guard


async def setup_streaming_stt(call_sid: str):
    """âš¡ Setup Deepgram streaming STT with improved VAD"""
    conn = manager.get(call_sid)
    if not conn:
        return

    try:
        dg_connection = deepgram.listen.live.v("1")

        def on_message(self, result, **kwargs):
            try:
                if not result or not result.channel:
                    return
                alt = result.channel.alternatives[0]
                transcript = alt.transcript
                if not transcript:
                    return

                is_final = result.is_final
                now = time.time()

                _logger.info("ðŸŽ™ï¸ STT %s: '%s'",
                             "FINAL" if is_final else "interim", transcript)

                # âœ… Always update speech time when we receive text
                conn.last_speech_time = now

                if is_final:
                    # ========================================
                    # âœ… FINAL RESULT - ALWAYS ACCUMULATE
                    # ========================================
                    current_buffer = conn.stt_transcript_buffer.strip()

                    if current_buffer:
                        # Check if this continues the current thought
                        if (not current_buffer.endswith(("^")) and
                                len(transcript) > 3):
                            # Continue the sentence
                            conn.stt_transcript_buffer += " " + transcript
                            _logger.info(
                                f"âž• Appending to sentence: '{transcript}'")
                        else:
                            # New thought or refinement
                            conn.stt_transcript_buffer = transcript
                            _logger.info(f"ðŸ”„ New sentence: '{transcript}'")
                    else:
                        # First content
                        conn.stt_transcript_buffer = transcript

                    # Mark that we have FINAL text
                    conn.stt_is_final = True

                    _logger.info(
                        f"ðŸ“ Complete buffer: '{conn.stt_transcript_buffer.strip()}'")

                else:
                    # ========================================
                    # âœ… INTERIM RESULT - TRACK BUT DON'T OVERWRITE
                    # ========================================

                    # Track interim time for activity detection
                    conn.last_interim_time = now
                    conn.last_interim_text = transcript

                    # Only use interim if we have no FINAL content yet
                    if not conn.stt_transcript_buffer or not conn.stt_is_final:
                        conn.stt_transcript_buffer = transcript
                        _logger.info(f"ðŸ“ Interim as buffer: '{transcript}'")

            except Exception as e:
                pass

        def on_open(self, open, **kwargs):
            pass

        def on_error(self, error, **kwargs):
            pass

        def on_close(self, close_msg, **kwargs):
            pass

        def on_speech_started(self, speech_started, **kwargs):
            """âœ… FIXED: Mark VAD trigger but require validation"""
            conn.vad_triggered_time = time.time()
            conn.user_speech_detected = True  # Tentatively set
            conn.speech_start_time = time.time()
            _logger.info("ðŸŽ¤ VAD: Speech trigger (needs validation)")

        def on_utterance_end(self, utterance_end, **kwargs):
            """âœ… FIXED: Clear VAD when Deepgram confirms utterance ended"""
            now = time.time()

            # Check if we got interim text very recently (within 200ms)
            if conn.last_interim_time and (now - conn.last_interim_time) < 0.2:
                _logger.info(
                    "â­ï¸ UtteranceEnd ignored - recent interim detected")
                return

            # âœ… Clear VAD state when Deepgram confirms end
            if conn.user_speech_detected:
                _logger.info(
                    "âœ… UtteranceEnd - clearing VAD (Deepgram confirmed)")
                conn.user_speech_detected = False
                conn.speech_start_time = None
                conn.vad_triggered_time = None
                conn.vad_validated = False
                conn.energy_drop_time = None

            conn.last_speech_time = now
            _logger.info(f"ðŸ•’ UtteranceEnd - last_speech_time: {now}")

        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(
            LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd,
                         on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        # Minimal, safe options for Twilio mu-law 8k (works on deepgram-sdk 3.2)
        options = LiveOptions(
            model=os.getenv("DEEPGRAM_STT_MODEL", "nova-2"),
            language="en-US",
            smart_format=True,
            interim_results=True,
            vad_events=True,
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            # If you want Deepgram to emit UtteranceEnd reliably, try enabling endpointing:
            # uncomment to try (if your project supports it)
            endpointing=UTTERANCE_END_MS,
        )

        # start() is synchronous and returns bool in SDK 3.2
        start_ok = False
        try:
            start_ok = dg_connection.start(options)
        except Exception as e:
            pass

        if not start_ok:
            fallback = LiveOptions(
                model=os.getenv("DEEPGRAM_STT_FALLBACK_MODEL",
                                "nova-2-general"),
                encoding="mulaw",
                sample_rate=8000,
                interim_results=True,
                # utterance_end_ms=UTTERANCE_END_MS,  # optional legacy param if endpointing not supported
            )
            try:
                start_ok = dg_connection.start(fallback)
            except Exception as e2:
                return

        if start_ok:
            conn.deepgram_live = dg_connection
            _logger.info("âœ… Streaming STT initialized")
        else:
            _logger.error(
                "âŒ Deepgram start() returned False (model/options/API key)")

    except Exception as e:
        pass


# ----------------------------

# âš¡ STREAMING PIPELINE
# ----------------------------
async def process_streaming_transcript(call_sid: str):
    """âœ… FIXED: Waits for COMPLETE final transcript + proper silence"""
    conn = manager.get(call_sid)
    if not conn or conn.is_responding:
        return

    if conn.interrupt_requested:
        _logger.debug("â­ï¸ Skipping - interrupt active")
        return

    now = time.time()

    # ========================================
    # âœ… CHECK 1: STUCK VAD TIMEOUT
    # ========================================
    if conn.user_speech_detected and conn.vad_triggered_time:
        vad_duration = now - conn.vad_triggered_time
        if vad_duration > conn.vad_timeout:
            _logger.warning(
                f"âš ï¸ Clearing stuck VAD (duration: {vad_duration:.1f}s)")
            conn.user_speech_detected = False
            conn.speech_start_time = None
            conn.vad_triggered_time = None
            conn.vad_validated = False

    # ========================================
    # âœ… CHECK 2: WAIT FOR USER TO FINISH SPEAKING
    # ========================================

    # Check if user is STILL speaking (recent interim text)
    if conn.last_interim_time and (now - conn.last_interim_time) < 0.5:
        _logger.debug(
            "â¸ï¸ User still adding to sentence (recent interim) - waiting...")
        return

    # If VAD says user is still speaking, wait
    if conn.user_speech_detected:
        _logger.debug("â¸ï¸ User still speaking (VAD active) - waiting...")
        return

    # ========================================
    # âœ… CHECK 3: MUST HAVE FINAL RESULT
    # ========================================

    # ðŸ”§ FIX: Must have at least ONE FINAL result before processing
    if not conn.stt_is_final:
        _logger.debug("â¸ï¸ Waiting for FINAL result...")
        return

    # ðŸ”§ FIX: Buffer must not be empty
    if not conn.stt_transcript_buffer or len(conn.stt_transcript_buffer.strip()) < 3:
        _logger.debug("â¸ï¸ Buffer empty or too short")
        return

    # ========================================
    # âœ… CHECK 4: ENFORCE SILENCE THRESHOLD
    # ========================================

    if not conn.last_speech_time:
        _logger.debug("â¸ï¸ No speech time recorded")
        return

    silence_elapsed = now - conn.last_speech_time

    # Enforce minimum silence
    if silence_elapsed < SILENCE_THRESHOLD_SEC:
        _logger.debug("â¸ï¸ Waiting for silence: %.2fs / %.1fs",
                      silence_elapsed, SILENCE_THRESHOLD_SEC)
        return

    # ========================================
    # âœ… CHECK 5: DOUBLE-CHECK FOR NEW SPEECH
    # ========================================

    # Small delay to catch any last-moment speech
    await asyncio.sleep(0.05)

    # Re-check after delay
    final_silence = time.time() - conn.last_speech_time

    # If new speech arrived during our checks, abort
    if final_silence < SILENCE_THRESHOLD_SEC:
        _logger.debug(
            "â¸ï¸ New speech detected during threshold check - resetting")
        return

    # Check if new interim/final arrived during our checks
    if conn.last_interim_time and (time.time() - conn.last_interim_time) < 0.3:
        _logger.debug("â¸ï¸ New interim detected - waiting for completion")
        return

    # ========================================
    # âœ… ALL CHECKS PASSED - PROCESS NOW
    # ========================================

    _logger.info("âœ… %.1fs silence threshold met (%.2fs)",
                 SILENCE_THRESHOLD_SEC, final_silence)

    # Mark as responding to prevent duplicate processing
    conn.is_responding = True

    try:
        # Get the COMPLETE accumulated transcript
        text = conn.stt_transcript_buffer.strip()

        # One final interrupt check
        if conn.interrupt_requested:
            _logger.debug("â­ï¸ Interrupt detected - aborting")
            conn.stt_transcript_buffer = ""
            conn.stt_is_final = False
            conn.last_interim_text = ""
            return

        # âœ… CHECK: Handle pending action confirmation
        if conn.pending_action:
            _logger.info(
                "Pending action detected. Checking user response: '%s'", text)
            confirmation = detect_confirmation_response(text)

            if confirmation == "yes":
                _logger.info("User confirmed action: %s",
                             conn.pending_action.get("tool"))
                result = await execute_detected_tool(call_sid, conn.pending_action)
                _logger.info("Confirmed tool execution result: %s", result)
                conn.pending_action = None
                conn.stt_transcript_buffer = ""
                conn.stt_is_final = False
                conn.last_interim_text = ""
                return
            elif confirmation == "no":
                await speak_text_streaming(call_sid, "Understood, I've cancelled that request. How else can I help you?")
                conn.pending_action = None
                conn.stt_transcript_buffer = ""
                conn.stt_is_final = False
                conn.last_interim_text = ""
                return
            else:
                await speak_text_streaming(call_sid, "Could you please confirm yes or no?")
                conn.stt_transcript_buffer = ""
                conn.stt_is_final = False
                conn.last_interim_text = ""
                return

        # âœ… CRITICAL: Clear buffer AFTER getting text
        conn.stt_transcript_buffer = ""
        conn.stt_is_final = False
        conn.last_interim_text = ""

        if not text or len(text) < 3:
            _logger.warning(f"⚠️ Text too short or empty: '{text}' - skipping")
            conn.is_responding = False
            return

        # Input: User transcript
        _logger.info(f"📝 USER INPUT: '{text}'")
        print(f"INPUT: {text}")

        t_start = time.time()

        # Stream LLM response
        response_buffer = ""
        sentence_buffer = ""
        sentence_count = 0
        MAX_SENTENCES = 10

        async for token in query_rag_streaming(text, conn.conversation_history, call_sid=call_sid):
            if conn.interrupt_requested:
                _logger.info("â­ï¸ Generation interrupted")
                break

            token = re.sub(r'\[(?:TOOL|CONFIRM_TOOL):[^\]]+\]', '', token)

            response_buffer += token
            sentence_buffer += token

            # Flush on sentence end
            if sentence_buffer.rstrip().endswith(('.', '?', '!')):
                sentence = sentence_buffer.strip()

                # ✅ Only queue non-empty sentences (skip if only had tool tags)
                if sentence:
                    sentence_count += 1
                    _logger.info("🎯 Sentence %d: '%s'",
                                 sentence_count, sentence)

                    # ✅ FIX: Handle queue full gracefully with backpressure
                    try:
                        await asyncio.wait_for(conn.tts_queue.put(sentence), timeout=2.0)
                    except asyncio.TimeoutError:
                        # If queue is full, skip this sentence to prevent deadlock
                        if conn.interrupt_requested:
                            break
                    except Exception as e:
                        if conn.interrupt_requested:
                            break

                sentence_buffer = ""

                if sentence_count >= MAX_SENTENCES:
                    break

        # Send any remaining text
        if not conn.interrupt_requested and sentence_buffer.strip():
            final_sentence = sentence_buffer.strip()
            # ✅ Only queue if not just tool tags
            if final_sentence and not re.match(r'^\s*\[\w+:[^\]]*\]\s*$', final_sentence):
                _logger.info("🎯 Final: '%s'", final_sentence)
                try:
                    await asyncio.wait_for(conn.tts_queue.put(final_sentence), timeout=2.0)
                except asyncio.TimeoutError:
                    _logger.warning("TTS queue full, dropping final sentence")
                except Exception as e:
                    _logger.error(f"Error queuing final sentence: {e}")

        # ✨ CRITICAL FIX: Save to conversation history IMMEDIATELY after LLM generates response
        # This ensures transcript is saved even if user hangs up during TTS playback
        cleaned_response, tool_data = parse_llm_response(response_buffer)
        
        if not conn.interrupt_requested and response_buffer.strip():
            conn.conversation_history.append({
                "user": text,
                "assistant": cleaned_response,
                "timestamp": time.time()
            })
            _logger.info(f"✅ Added to conversation_history BEFORE TTS: user='{text[:50]}...', assistant='{cleaned_response[:50]}...'")
            _logger.info(f"   Total history entries: {len(conn.conversation_history)}")

            # Keep last 10 exchanges
            if len(conn.conversation_history) > 10:
                conn.conversation_history = conn.conversation_history[-10:]
        else:
            _logger.warning(f"⚠️ NOT added to history - interrupt: {conn.interrupt_requested}, response empty: {not response_buffer.strip()}")

        # Handle tool calls
        if tool_data:
            _logger.info("Tool detected: %s - requires_confirmation: %s",
                         tool_data.get('tool'), tool_data.get('requires_confirmation'))

            if tool_data.get("requires_confirmation"):
                conn.pending_action = tool_data
                _logger.info("â³ Awaiting user confirmation for: %s",
                             tool_data.get("tool"))
            else:
                result = await execute_detected_tool(call_sid, tool_data)
                _logger.info("Tool execution result: %s", result)

        _logger.info("⏳ Waiting for TTS...")
        # Wait for TTS queue to empty (all sentences spoken)
        max_wait = 30.0
        wait_start = time.time()
        while not conn.tts_queue.empty() and (time.time() - wait_start) < max_wait:
            await asyncio.sleep(0.1)
        _logger.info("✅ TTS completed")

        t_end = time.time()
        _logger.info("âœ… TOTAL PROCESSING TIME: %.1fms",
                     (t_end - t_start) * 1000)

    except Exception as e:
        # ✨ FIX: Log errors instead of silently ignoring them!
        _logger.error(f"❌ ERROR in process_streaming_transcript: {e}")
        import traceback
        _logger.error(traceback.format_exc())
        
        # ✨ Still try to save what we have to conversation history
        if 'text' in locals() and 'response_buffer' in locals() and response_buffer:
            try:
                conn.conversation_history.append({
                    "user": text,
                    "assistant": f"[Error: {str(e)[:100]}] {response_buffer[:200]}",
                    "timestamp": time.time()
                })
                _logger.info(f"✅ Saved partial response to history despite error")
            except:
                pass
    finally:
        conn.is_responding = False
        if conn.interrupt_requested:
            conn.interrupt_requested = False


# FastAPI app
# ----------------------------
app = FastAPI(
    title="AI Voice Call System - ElevenLabs Compatible",
    description="Self-hosted voice AI with agent management, webhooks, and dynamic variables",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# HELPER FUNCTIONS
# ================================

def generate_agent_id() -> str:
    """Generate unique agent ID"""
    return f"agent_{uuid.uuid4().hex[:16]}"


def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    return f"conv_{uuid.uuid4().hex[:16]}"


async def send_webhook(webhook_url: str, event: str, data: Dict):
    """Send webhook notification to registered webhook URLs (fire-and-forget)"""
    try:
        # Webhook URL must be absolute (http:// or https://)
        if not webhook_url.startswith(("http://", "https://")):
            _logger.error(f"❌ Invalid webhook URL: {webhook_url} - must start with http:// or https://")
            return False
        
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event,
                "timestamp": dt.utcnow().isoformat(),
                "data": data
            }
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            _logger.info(f"📤 Webhook sent: {event} to {webhook_url} (status: {response.status_code})")
            return response.status_code == 200
    except Exception as e:
        _logger.error(f"❌ Webhook failed: {event} to {webhook_url} - {e}")
        return False


async def send_webhook_and_get_response(webhook_url: str, event: str, data: Dict) -> Optional[Dict]:
    """Send webhook and wait for response data (for inbound call configuration)"""
    try:
        # Webhook URL must be absolute (http:// or https://)
        if not webhook_url.startswith(("http://", "https://")):
            _logger.error(f"❌ Invalid webhook URL: {webhook_url} - must start with http:// or https://")
            return None
        
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event,
                "timestamp": dt.utcnow().isoformat(),
                "data": data
            }
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            _logger.info(f"📤 Webhook sent: {event} to {webhook_url} (status: {response.status_code})")
            
            if response.status_code == 200:
                response_data = response.json()
                _logger.info(f"📥 Webhook response received: {list(response_data.keys())}")
                return response_data
            else:
                _logger.warning(f"⚠️ Webhook returned non-200 status: {response.status_code}")
                return None
    except Exception as e:
        _logger.error(f"❌ Webhook failed: {event} to {webhook_url} - {e}")
        return None


# ================================
# AGENT MANAGEMENT API
# ================================

@app.post("/v1/convai/agents", tags=["Agent Management"])
async def create_agent(
    agent: AgentCreate, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new agent with custom configuration
    
    Like ElevenLabs: Each agent has system prompt, voice, model settings
    """
    try:
        agent_id = generate_agent_id()
        
        db_agent = Agent(
            agent_id=agent_id,
            name=agent.name,
            system_prompt=agent.system_prompt,
            first_message=agent.first_message,
            voice_provider=agent.voice_provider,
            voice_id=agent.voice_id,
            model_provider=agent.model_provider,
            model_name=agent.model_name,
            interrupt_enabled=agent.interrupt_enabled,
            silence_threshold_sec=agent.silence_threshold_sec
        )
        
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        
        _logger.info(f"✅ Created agent: {agent_id} - {agent.name}")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "name": agent.name,
            "created_at": db_agent.created_at.isoformat()
        }
    except Exception as e:
        db.rollback()
        _logger.error(f"❌ Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/convai/agents/{agent_id}", tags=["Agent Management"])
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """Get agent configuration"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "system_prompt": agent.system_prompt,
        "first_message": agent.first_message,
        "voice_provider": agent.voice_provider,
        "voice_id": agent.voice_id,
        "model_provider": agent.model_provider,
        "model_name": agent.model_name,
        "interrupt_enabled": agent.interrupt_enabled,
        "silence_threshold_sec": agent.silence_threshold_sec,
        "is_active": agent.is_active,
        "created_at": agent.created_at.isoformat(),
        "updated_at": agent.updated_at.isoformat()
    }


@app.get("/v1/convai/agents", tags=["Agent Management"])
async def list_agents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all agents"""
    agents = db.query(Agent).filter(Agent.is_active == True).offset(skip).limit(limit).all()
    
    return {
        "agents": [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "voice_id": agent.voice_id,
                "model_name": agent.model_name,
                "created_at": agent.created_at.isoformat()
            }
            for agent in agents
        ],
        "total": len(agents)
    }


@app.patch("/v1/convai/agents/{agent_id}", tags=["Agent Management"])
async def update_agent(
    agent_id: str,
    updates: AgentUpdate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Update agent configuration"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update only provided fields
    update_data = updates.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(agent, key, value)
    
    agent.updated_at = dt.utcnow()
    db.commit()
    db.refresh(agent)
    
    _logger.info(f"✅ Updated agent: {agent_id}")
    
    return {
        "success": True,
        "agent_id": agent_id,
        "updated_fields": list(update_data.keys())
    }


@app.delete("/v1/convai/agents/{agent_id}", tags=["Agent Management"])
async def delete_agent(
    agent_id: str, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete (deactivate) agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.is_active = False
    db.commit()
    
    _logger.info(f"✅ Deleted agent: {agent_id}")
    
    return {"success": True, "message": "Agent deleted"}


# ================================
# ELEVENLABS-COMPATIBLE CALL API
# ================================

@app.post("/v1/convai/twilio/outbound-call", tags=["Call Operations"])
async def initiate_outbound_call(
    request: OutboundCallRequest,
    db: Session = Depends(get_db)
):
    """
    ✨ ELEVENLABS-COMPATIBLE ENDPOINT
    
    Initiate outbound call with agent configuration and dynamic variables
    
    Request format (matches ElevenLabs):
    {
        "agent_id": "agent_xxx",
        "to_number": "+1234567890",
        "conversation_initiation_client_data": {
            "dynamic_variables": {
                "customer_name": "John",
                "company": "Acme Corp",
                ...
            },
            "conversation_config_override": {
                "tts": {"voice_id": "custom_voice"},
                "agent": {"prompt": {"llm": "custom_model"}}
            }
        }
    }
    """
    try:
        # Validate agent exists
        agent = db.query(Agent).filter(
            Agent.agent_id == request.agent_id,
            Agent.is_active == True
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_id}")
        
        # Extract dynamic variables and overrides
        client_data = request.conversation_initiation_client_data or {}
        dynamic_variables = client_data.get("dynamic_variables", {})
        config_override = client_data.get("conversation_config_override", {})
        
        # 🔍 DEBUG: Log raw extraction
        _logger.info(f"🔍 API Debug - client_data keys: {list(client_data.keys())}")
        _logger.info(f"🔍 API Debug - config_override: {config_override}")
        _logger.info(f"🔍 API Debug - tts section: {config_override.get('tts', {})}")
        
        # Extract voice and model overrides
        custom_voice_id = config_override.get("tts", {}).get("voice_id")
        custom_model = config_override.get("agent", {}).get("prompt", {}).get("llm")
        custom_first_message = request.first_message or config_override.get("agent", {}).get("first_message")

        _logger.info(f"📞 Initiating call to {request.to_number} with agent {request.agent_id}")
        _logger.info(f"📊 Dynamic variables: {len(dynamic_variables)} fields")
        
        # 🔍 DEBUG: Log extracted values
        _logger.info(f"🔍 API Extracted - custom_voice_id: '{custom_voice_id}'")
        _logger.info(f"🔍 API Extracted - custom_model: '{custom_model}'")
        _logger.info(f"🔍 API Extracted - custom_first_message: '{custom_first_message[:50] if custom_first_message else None}...'")

        if custom_voice_id:
            _logger.info(f"🎤 Voice override: {custom_voice_id}")
        if custom_model:
            _logger.info(f"🤖 Model override: {custom_model}")
        
        # ✨ Look up phone number from database (priority order)
        phone_number_to_use = TWILIO_PHONE_NUMBER  # Default fallback from env
        
        # Priority 1: Use agent_phone_number_id from request (if provided)
        if request.agent_phone_number_id:
            phone_record = db.query(PhoneNumber).filter(
                PhoneNumber.id == request.agent_phone_number_id,
                PhoneNumber.is_active == True
            ).first()
            if phone_record:
                phone_number_to_use = phone_record.phone_number
                _logger.info(f"📞 Using phone number from database (ID: {request.agent_phone_number_id}): {phone_number_to_use}")
            else:
                _logger.warning(f"⚠️ Phone number ID '{request.agent_phone_number_id}' not found in database, using fallback")
        
        # Priority 2: Try to get phone number linked to agent
        if phone_number_to_use == TWILIO_PHONE_NUMBER and agent.agent_id:
            phone_record = db.query(PhoneNumber).filter(
                PhoneNumber.agent_id == agent.agent_id,
                PhoneNumber.is_active == True
            ).first()
            if phone_record:
                phone_number_to_use = phone_record.phone_number
                _logger.info(f"📞 Using agent's linked phone number: {phone_number_to_use}")
        
        # Priority 3: Use TWILIO_PHONE_NUMBER from env (already set as default)
        if phone_number_to_use == TWILIO_PHONE_NUMBER:
            _logger.info(f"📞 Using default phone number from env: {phone_number_to_use}")
        
        # Make Twilio call
        webhook_url = f"{PUBLIC_URL.rstrip('/')}/voice/outbound"
        status_callback_url = f"{PUBLIC_URL.rstrip('/')}/voice/status"
        
        call = twilio_client.calls.create(
            to=request.to_number,
            from_=phone_number_to_use,  # ✅ From database lookup or env fallback
            url=webhook_url,
            method="POST",
            status_callback=status_callback_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )
        
        call_sid = call.sid
        conversation_id = call_sid  # Use Twilio call_sid as conversation_id
        
        # Store call data for when WebSocket connects
        pending_call_data[call_sid] = {
            "agent_id": request.agent_id,
            "dynamic_variables": dynamic_variables,
            "custom_voice_id": custom_voice_id,
            "custom_model": custom_model,
            "custom_first_message": custom_first_message,
            "to_number": request.to_number,
            "enable_recording": request.enable_recording,
            "direction": "outbound"
        }
        
        _logger.info(f"💾 Stored call data for: {call_sid}")
        _logger.info(f"💾 - Agent ID: {request.agent_id}")
        _logger.info(f"💾 - Custom voice: {custom_voice_id}")
        _logger.info(f"💾 - Custom model: {custom_model}")
        _logger.info(f"💾 - Dynamic vars: {len(dynamic_variables)} fields")
        
        # Create conversation record in database
        conversation = Conversation(
            conversation_id=conversation_id,
            agent_id=request.agent_id,
            phone_number=request.to_number,
            status="initiated",
            dynamic_variables=dynamic_variables,
            call_metadata={"overrides": config_override,
            "custom_first_message": custom_first_message}
        )
        db.add(conversation)
        db.commit()
        
        _logger.info(f"✅ Call initiated: {conversation_id}")
        
        # Send webhook notification (if configured)
        webhooks = db.query(WebhookConfig).filter(
            WebhookConfig.is_active == True
        ).filter(
            (WebhookConfig.agent_id == request.agent_id) | (WebhookConfig.agent_id == None)
        ).all()
        
        for webhook in webhooks:
            if "call.initiated" in webhook.events or not webhook.events:
                await send_webhook(
                    webhook.webhook_url,
                    "call.initiated",
                    {
                        "conversation_id": conversation_id,
                        "agent_id": request.agent_id,
                        "to_number": request.to_number,
                        "status": "initiated"
                    }
                )
        
        # Return ElevenLabs-compatible response
        return {
            "conversation_id": conversation_id,
            "agent_id": request.agent_id,
            "status": "initiated",
            "phone_number": request.to_number,
            "first_message": custom_first_message or agent.first_message  
        }
        
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"❌ Call initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# CONVERSATION RETRIEVAL API
# ================================

@app.get("/v1/convai/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """
    ✨ ELEVENLABS-COMPATIBLE ENDPOINT
    
    Get conversation details including transcript
    """
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Extract direction from call_metadata
    call_direction = "outbound"
    if conversation.call_metadata and isinstance(conversation.call_metadata, dict):
        call_direction = conversation.call_metadata.get("direction", "outbound")
    
    # Get agent name if exists
    agent_name = None
    if conversation.agent_id:
        agent = db.query(Agent).filter(Agent.agent_id == conversation.agent_id).first()
        if agent:
            agent_name = agent.name
    
    return {
        "conversation_id": conversation.conversation_id,
        "agent_id": conversation.agent_id,
        "agent_name": agent_name,
        "status": conversation.status,
        "transcript": conversation.transcript,
        "started_at": conversation.started_at.isoformat() if conversation.started_at else None,
        "ended_at": conversation.ended_at.isoformat() if conversation.ended_at else None,
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
        "metadata": {
            "call_duration_secs": conversation.duration_secs,
            "termination_reason": conversation.ended_reason,
            "phone_number": conversation.phone_number,
            "direction": call_direction,
            "recording_url": conversation.recording_url
        },
        "analysis": {
            "transcript_length": len(conversation.transcript) if conversation.transcript else 0,
            "has_recording": bool(conversation.recording_url)
        },
        "dynamic_variables": conversation.dynamic_variables,
        "call_metadata": conversation.call_metadata
    }


@app.get("/v1/convai/conversations", tags=["Conversations"])
async def list_conversations(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    direction: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List conversations (optionally filtered by agent_id, status, direction)
    
    ✨ ELEVENLABS-COMPATIBLE: Supports filtering and pagination
    """
    query = db.query(Conversation)
    
    if agent_id:
        query = query.filter(Conversation.agent_id == agent_id)
    
    if status:
        query = query.filter(Conversation.status == status)
    
    conversations = query.order_by(Conversation.created_at.desc()).offset(skip).limit(limit).all()
    
    # Filter by direction if specified (direction is in call_metadata)
    if direction:
        conversations = [
            conv for conv in conversations
            if conv.call_metadata and isinstance(conv.call_metadata, dict) 
            and conv.call_metadata.get("direction") == direction
        ]
    
    # Get total count (without filters for pagination info)
    total_query = db.query(Conversation)
    if agent_id:
        total_query = total_query.filter(Conversation.agent_id == agent_id)
    if status:
        total_query = total_query.filter(Conversation.status == status)
    total_count = total_query.count()
    
    return {
        "conversations": [
            {
                "conversation_id": conv.conversation_id,
                "agent_id": conv.agent_id,
                "status": conv.status,
                "phone_number": conv.phone_number,
                "duration_secs": conv.duration_secs,
                "direction": conv.call_metadata.get("direction", "outbound") if conv.call_metadata and isinstance(conv.call_metadata, dict) else "outbound",
                "ended_reason": conv.ended_reason,
                "has_transcript": bool(conv.transcript),
                "has_recording": bool(conv.recording_url),
                "started_at": conv.started_at.isoformat() if conv.started_at else None,
                "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                "created_at": conv.created_at.isoformat() if conv.created_at else None
            }
            for conv in conversations
        ],
        "total": total_count,
        "page_size": limit,
        "offset": skip
    }


# ================================
# WEBHOOK MANAGEMENT API
# ================================

@app.post("/v1/convai/webhooks", tags=["Webhooks"], response_model=WebhookResponse)
async def create_webhook(
    request: WebhookCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Register webhook for call events
    
    **Available Events:**
    - `call.initiated` - When a call is initiated
    - `call.started` - When a call connects
    - `call.ended` - When a call ends
    - `call.failed` - When a call fails
    - `transcript.partial` - Partial transcript updates
    - `transcript.final` - Final transcript
    - `agent.response` - Agent responds
    - `tool.called` - When a tool is called
    - `user.interrupted` - When user interrupts
    
    **Examples:**
    ```json
    {
      "webhook_url": "https://your-app.com/webhook",
      "events": ["call.started", "call.ended"],
      "agent_id": "agent_123"
    }
    ```
    
    Set `agent_id` to `null` for global webhooks (all agents).
    """
    try:
        # Validate webhook URL
        if not request.webhook_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400, 
                detail="Webhook URL must start with http:// or https://"
            )
        
        # Validate events
        if request.events:
            invalid_events = [e for e in request.events if e not in WEBHOOK_EVENTS]
            if invalid_events:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid events: {invalid_events}. Valid events: {WEBHOOK_EVENTS}"
                )
        
        # If agent_id provided, verify agent exists
        if request.agent_id:
            agent = db.query(Agent).filter(Agent.agent_id == request.agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_id}")
        
        webhook = WebhookConfig(
            agent_id=request.agent_id,
            webhook_url=request.webhook_url,
            events=request.events or WEBHOOK_EVENTS
        )
        
        db.add(webhook)
        db.commit()
        db.refresh(webhook)
        
        _logger.info(
            f"✅ Webhook registered: {request.webhook_url} "
            f"for agent: {request.agent_id or 'GLOBAL'} "
            f"with events: {request.events}"
        )
        
        return WebhookResponse(
            success=True,
            webhook_id=webhook.id,
            webhook_url=request.webhook_url,
            events=webhook.events,
            agent_id=request.agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"❌ Webhook creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/convai/webhooks", tags=["Webhooks"])
async def list_webhooks(
    agent_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all registered webhooks
    
    **Query Parameters:**
    - `agent_id` (optional): Filter by agent ID. Omit to see all webhooks.
    
    **Returns:**
    - List of webhooks with their configuration
    - Includes global webhooks (agent_id = null)
    """
    query = db.query(WebhookConfig).filter(WebhookConfig.is_active == True)
    
    if agent_id:
        query = query.filter(WebhookConfig.agent_id == agent_id)
    
    webhooks = query.all()
    
    return {
        "webhooks": [
            {
                "id": w.id,
                "agent_id": w.agent_id or "GLOBAL",
                "webhook_url": w.webhook_url,
                "events": w.events,
                "created_at": w.created_at.isoformat() if w.created_at else None
            }
            for w in webhooks
        ],
        "total": len(webhooks)
    }


@app.delete("/v1/convai/webhooks/{webhook_id}", tags=["Webhooks"])
async def delete_webhook(
    webhook_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete a webhook by ID
    
    **Path Parameters:**
    - `webhook_id`: The numeric ID of the webhook to delete
    
    **Returns:**
    - Success confirmation
    """
    webhook = db.query(WebhookConfig).filter(WebhookConfig.id == webhook_id).first()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_url = webhook.webhook_url
    agent_id = webhook.agent_id
    
    webhook.is_active = False
    db.commit()
    
    _logger.info(f"✅ Webhook deleted: ID={webhook_id}, URL={webhook_url}, Agent={agent_id or 'GLOBAL'}")
    
    return {
        "success": True,
        "message": "Webhook deleted successfully",
        "webhook_id": webhook_id
    }


# ================================
# PHONE NUMBER MANAGEMENT API
# ================================

@app.post("/v1/convai/phone-numbers", tags=["Phone Numbers"])
async def register_phone_number(
    phone_number: str,
    agent_id: Optional[str] = None,
    label: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Register a phone number and optionally link to agent"""
    # Check if phone number already exists
    existing = db.query(PhoneNumber).filter(
        PhoneNumber.phone_number == phone_number
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    # Verify agent exists if provided
    if agent_id:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
    
    phone = PhoneNumber(
        id=f"pn_{uuid.uuid4().hex[:16]}",
        phone_number=phone_number,
        agent_id=agent_id,
        label=label
    )
    db.add(phone)
    db.commit()
    db.refresh(phone)
    
    _logger.info(f"✅ Registered phone number: {phone_number} -> agent: {agent_id}")
    
    return {
        "phone_number_id": phone.id,
        "phone_number": phone_number,
        "agent_id": agent_id,
        "label": label
    }


@app.get("/v1/convai/phone-numbers", tags=["Phone Numbers"])
async def list_phone_numbers(db: Session = Depends(get_db)):
    """List all registered phone numbers"""
    phones = db.query(PhoneNumber).filter(PhoneNumber.is_active == True).all()
    
    return {
        "phone_numbers": [
            {
                "id": p.id,
                "phone_number": p.phone_number,
                "agent_id": p.agent_id,
                "label": p.label,
                "provider": p.provider,
                "created_at": p.created_at.isoformat()
            }
            for p in phones
        ]
    }


@app.patch("/v1/convai/phone-numbers/{phone_id}", tags=["Phone Numbers"])
async def update_phone_number(
    phone_id: str,
    agent_id: Optional[str] = None,
    label: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Update phone number configuration (link to different agent)"""
    phone = db.query(PhoneNumber).filter(PhoneNumber.id == phone_id).first()
    
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    if agent_id is not None:
        if agent_id:  # Not empty string
            agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        phone.agent_id = agent_id if agent_id else None
    
    if label is not None:
        phone.label = label
    
    db.commit()
    
    return {"success": True, "phone_number_id": phone_id}


@app.delete("/v1/convai/phone-numbers/{phone_id}", tags=["Phone Numbers"])
async def delete_phone_number(
    phone_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a phone number"""
    phone = db.query(PhoneNumber).filter(PhoneNumber.id == phone_id).first()
    
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    phone.is_active = False
    db.commit()
    
    return {"success": True, "message": "Phone number deleted"}


# ================================
# KNOWLEDGE BASE PER AGENT API
# ================================

@app.post("/v1/convai/agents/{agent_id}/knowledge-base", tags=["Knowledge Base"])
async def add_knowledge(
    agent_id: str,
    content: str,
    metadata: Optional[Dict] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Add knowledge to agent's knowledge base"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    doc_id = f"doc_{uuid.uuid4().hex[:16]}"
    
    # Add to database
    kb = KnowledgeBase(
        agent_id=agent_id,
        document_id=doc_id,
        content=content,
        kb_metadata=metadata
    )
    db.add(kb)
    db.commit()
    
    # Add to ChromaDB with agent prefix
    chunks = _chunk_text(content, CHUNK_SIZE, overlap=50)
    
    with torch.no_grad():
        embeddings = embedder.encode(
            chunks, 
            device=DEVICE, 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
    
    # Use agent-specific collection
    agent_collection = chroma_client.get_or_create_collection(f"agent_{agent_id}")
    
    agent_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"agent_id": agent_id, "doc_id": doc_id} for _ in chunks]
    )
    
    _logger.info(f"✅ Added knowledge to agent {agent_id}: {len(chunks)} chunks")
    
    return {
        "document_id": doc_id,
        "agent_id": agent_id,
        "chunks_created": len(chunks)
    }


@app.get("/v1/convai/agents/{agent_id}/knowledge-base", tags=["Knowledge Base"])
async def list_agent_knowledge(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """List knowledge base documents for an agent"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    documents = db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id
    ).all()
    
    return {
        "agent_id": agent_id,
        "documents": [
            {
                "document_id": doc.document_id,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.kb_metadata,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ],
        "total": len(documents)
    }


@app.delete("/v1/convai/agents/{agent_id}/knowledge-base/{document_id}", tags=["Knowledge Base"])
async def delete_agent_knowledge(
    agent_id: str,
    document_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a knowledge base document"""
    doc = db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id,
        KnowledgeBase.document_id == document_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from database
    db.delete(doc)
    db.commit()
    
    # Remove from ChromaDB
    try:
        agent_collection = chroma_client.get_or_create_collection(f"agent_{agent_id}")
        # Get all IDs that start with this document_id
        results = agent_collection.get(where={"doc_id": document_id})
        if results and results.get("ids"):
            agent_collection.delete(ids=results["ids"])
    except Exception as e:
        _logger.warning(f"⚠️ Could not delete from ChromaDB: {e}")
    
    return {"success": True, "message": "Document deleted"}


# ================================
# CUSTOM TOOLS PER AGENT API
# ================================

@app.post("/v1/convai/agents/{agent_id}/tools", tags=["Tools"])
async def add_agent_tool(
    agent_id: str,
    tool_name: str,
    description: str,
    webhook_url: Optional[str] = None,
    parameters: Dict = {},
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Add custom tool to agent"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    tool = AgentTool(
        agent_id=agent_id,
        tool_name=tool_name,
        description=description,
        webhook_url=webhook_url,
        parameters=parameters
    )
    db.add(tool)
    db.commit()
    db.refresh(tool)
    
    _logger.info(f"✅ Added tool '{tool_name}' to agent {agent_id}")
    
    return {
        "success": True,
        "tool_id": tool.id,
        "tool_name": tool_name,
        "agent_id": agent_id
    }


@app.get("/v1/convai/agents/{agent_id}/tools", tags=["Tools"])
async def list_agent_tools(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """List custom tools for an agent"""
    tools = db.query(AgentTool).filter(
        AgentTool.agent_id == agent_id,
        AgentTool.is_active == True
    ).all()
    
    return {
        "agent_id": agent_id,
        "tools": [
            {
                "id": t.id,
                "tool_name": t.tool_name,
                "description": t.description,
                "webhook_url": t.webhook_url,
                "parameters": t.parameters,
                "created_at": t.created_at.isoformat()
            }
            for t in tools
        ]
    }


@app.delete("/v1/convai/agents/{agent_id}/tools/{tool_id}", tags=["Tools"])
async def delete_agent_tool(
    agent_id: str,
    tool_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a custom tool"""
    tool = db.query(AgentTool).filter(
        AgentTool.id == tool_id,
        AgentTool.agent_id == agent_id
    ).first()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool.is_active = False
    db.commit()
    
    return {"success": True, "message": "Tool deleted"}


# ================================
# CALL RECORDING API
# ================================

@app.post("/recording-callback", tags=["Recording"])
async def recording_callback(request: Request):
    """Handle recording completion from Twilio"""
    form = await request.form()
    call_sid = form.get("CallSid")
    recording_url = form.get("RecordingUrl")
    recording_sid = form.get("RecordingSid")
    recording_duration = form.get("RecordingDuration")
    
    _logger.info(f"🎙️ Recording completed: {call_sid} - {recording_url}")
    
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            conversation.recording_url = recording_url
            # Store additional recording metadata
            if conversation.call_metadata:
                conversation.call_metadata["recording_sid"] = recording_sid
                conversation.call_metadata["recording_duration"] = recording_duration
            else:
                conversation.call_metadata = {
                    "recording_sid": recording_sid,
                    "recording_duration": recording_duration
                }
            db.commit()
            _logger.info(f"✅ Recording URL saved for {call_sid}")
    except Exception as e:
        _logger.error(f"❌ Failed to save recording URL: {e}")
    finally:
        db.close()
    
    return PlainTextResponse("OK")


@app.get("/v1/convai/conversations/{conversation_id}/recording", tags=["Recording"])
async def get_recording(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get recording URL for a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if not conversation.recording_url:
        raise HTTPException(status_code=404, detail="No recording available")
    
    return {
        "conversation_id": conversation_id,
        "recording_url": conversation.recording_url,
        "recording_metadata": conversation.call_metadata
    }


# ================================
# SIGNED URL FOR WIDGETS (JWT)
# ================================

@app.get("/v1/convai/conversation/get-signed-url", tags=["Widgets"])
async def get_signed_url(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """Generate signed URL for embedding widget"""
    import jwt
    from datetime import timedelta
    
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    payload = {
        "agent_id": agent_id,
        "exp": dt.utcnow() + timedelta(hours=24),
        "iat": dt.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    return {
        "signed_url": f"{PUBLIC_URL}/widget?token={token}",
        "expires_in": 86400,  # 24 hours in seconds
        "agent_id": agent_id
    }


@app.get("/widget", tags=["Widgets"])
async def widget_page(
    token: str,
    db: Session = Depends(get_db)
):
    """Widget endpoint that validates JWT token"""
    import jwt
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        agent_id = payload.get("agent_id")
        
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "valid": True,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "message": "Widget authentication successful"
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/test-end-call")
async def test_end_call(request: Request):
    """Test end call tool"""
    try:
        data = await request.json()
        call_sid = data.get("call_sid", "test_call_123")
        reason = data.get("reason", "test")

        result = await end_call_tool(call_sid, reason)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/test-transfer")
async def test_transfer(request: Request):
    """Test transfer tool"""
    try:
        data = await request.json()
        call_sid = data.get("call_sid", "test_call_123")
        department = data.get("department", "sales")

        result = await transfer_call_tool(call_sid, department)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/tools/status")
async def tools_status():
    """Check tool configuration"""
    return {
        "tools_available": ["end_call", "transfer_call"],
        "departments": {
            "sales": os.getenv("SALES_PHONE_NUMBER", "NOT_SET"),
            "support": os.getenv("SUPPORT_PHONE_NUMBER", "NOT_SET"),
            "technical": os.getenv("TECH_PHONE_NUMBER", "NOT_SET"),
        },
        "confirmation_system": "enabled",
        "transfer_requires_confirmation": True,
        "end_call_requires_confirmation": False,
        "silence_threshold_sec": SILENCE_THRESHOLD_SEC,
        "utterance_end_ms": UTTERANCE_END_MS


    }


@app.websocket("/media-stream")
async def media_ws(websocket: WebSocket):
    try:
        await websocket.accept()
    except RuntimeError as e:
        return

    async def send_heartbeat():
        while True:
            try:
                await asyncio.sleep(5)
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"event": "heartbeat"})
            except Exception as e:
                break

    heartbeat_task = asyncio.create_task(send_heartbeat())

    current_call_sid: Optional[str] = None

    processing_task: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except RuntimeError as e:
                break
            except Exception as e:
                break

            event = data.get("event")

            if event == "start":
                start_info = data.get("start", {})
                current_call_sid = start_info.get("callSid")
                stream_sid = start_info.get("streamSid")

                if not current_call_sid:
                    break

                await manager.connect(current_call_sid, websocket)
                conn = manager.get(current_call_sid)
                if conn:
                    conn.stream_sid = stream_sid
                    conn.stream_ready = True
                    conn.conversation_id = current_call_sid

                    # ✨ Load agent configuration and call data
                    call_data = pending_call_data.get(current_call_sid, {})
                    agent_id = call_data.get("agent_id")
                    call_direction = call_data.get("direction", "outbound")
                    
                    _logger.info(f"🔍 WebSocket Debug - call_sid: {current_call_sid}")
                    _logger.info(f"🔍 Pending call data found: {bool(call_data)}")
                    _logger.info(f"🔍 Agent ID: {agent_id}")
                    _logger.info(f"🔍 Direction: {call_direction}")
                    _logger.info(f"🔍 Custom voice_id: {call_data.get('custom_voice_id')}")
                    _logger.info(f"🔍 Custom model: {call_data.get('custom_model')}")
                    
                    # ✨ ALWAYS load dynamic variables (like ElevenLabs)
                    conn.dynamic_variables = call_data.get("dynamic_variables", {})
                    conn.custom_voice_id = call_data.get("custom_voice_id")
                    conn.custom_model = call_data.get("custom_model")
                    conn.custom_first_message = call_data.get("custom_first_message")
                    
                    # ✨ Log all overrides for debugging
                    _logger.info(f"🔧 Overrides loaded:")
                    _logger.info(f"   - custom_voice_id: {conn.custom_voice_id or 'None (will use agent/default)'}")
                    _logger.info(f"   - custom_model: {conn.custom_model or 'None (will use agent/default)'}")
                    _logger.info(f"   - custom_first_message: {'Yes (' + conn.custom_first_message[:30] + '...)' if conn.custom_first_message else 'None (will use agent/default)'}")
                    
                    db = SessionLocal()
                    try:
                        # Load agent if specified
                        if agent_id:
                            agent = db.query(Agent).filter(
                                Agent.agent_id == agent_id
                            ).first()
                            
                            if agent:
                                conn.agent_id = agent_id
                                conn.agent_config = {
                                    "system_prompt": agent.system_prompt,
                                    "first_message": agent.first_message,
                                    "voice_id": agent.voice_id,
                                    "model_name": agent.model_name,
                                    "silence_threshold_sec": agent.silence_threshold_sec
                                }
                                
                                _logger.info(f"✅ Loaded agent: {agent_id} ({agent.name})")
                                _logger.info(f"📊 Dynamic variables: {len(conn.dynamic_variables)} fields")

                                if call_data.get("custom_first_message"):
                                    conn.agent_config["first_message"] = call_data["custom_first_message"]
                                    _logger.info(f"💬 Using custom first message: {call_data['custom_first_message'][:50]}...")
                            else:
                                _logger.warning(f"⚠️ Agent not found: {agent_id}")
                        else:
                            _logger.info("ℹ️ No agent specified, using default behavior")
                        
                        # ✨ ALWAYS update conversation status to "in-progress" (like ElevenLabs)
                        conversation = db.query(Conversation).filter(
                            Conversation.conversation_id == current_call_sid
                        ).first()
                        
                        if conversation:
                            conversation.status = "in-progress"
                            conversation.started_at = dt.utcnow()
                            db.commit()
                            _logger.info(f"✅ Conversation status updated to 'in-progress': {current_call_sid}")
                        else:
                            # Create conversation record if it doesn't exist (fallback)
                            _logger.warning(f"⚠️ Conversation not found, creating new record: {current_call_sid}")
                            # ✅ For inbound: use from_number (caller), for outbound: use to_number (recipient)
                            phone_for_record = call_data.get("from_number") if call_direction == "inbound" else call_data.get("to_number")
                            new_conversation = Conversation(
                                conversation_id=current_call_sid,
                                agent_id=agent_id,
                                phone_number=phone_for_record,
                                status="in-progress",
                                started_at=dt.utcnow(),
                                dynamic_variables=conn.dynamic_variables,
                                call_metadata={"direction": call_direction}
                            )
                            db.add(new_conversation)
                            db.commit()
                        
                        # ✨ ALWAYS send "call.started" webhook (like ElevenLabs)
                        webhooks = db.query(WebhookConfig).filter(
                            WebhookConfig.is_active == True
                        ).all()
                        
                        for webhook in webhooks:
                            should_send = False
                            if webhook.agent_id is None:
                                should_send = True  # Global webhook
                            elif agent_id and webhook.agent_id == agent_id:
                                should_send = True  # Agent-specific webhook
                            
                            if should_send and ("call.started" in webhook.events or not webhook.events):
                                # ✅ For inbound: send caller's number (from_number in call_data)
                                # ✅ For outbound: send recipient's number (to_number in call_data)
                                caller_phone = call_data.get("from_number") if call_direction == "inbound" else call_data.get("to_number")
                                
                                # ✅ For INBOUND calls: Wait for webhook response to get dynamic variables
                                if call_direction == "inbound":
                                    _logger.info(f"🔄 Sending call.started webhook to {webhook.webhook_url} and waiting for response...")
                                    webhook_response = await send_webhook_and_get_response(
                                        webhook.webhook_url,
                                        "call.started",
                                        {
                                            "conversation_id": current_call_sid,
                                            "agent_id": agent_id,
                                            "direction": call_direction,
                                            "status": "in-progress",
                                            "phone_number": caller_phone
                                        }
                                    )
                                    
                                    _logger.info(f"📥 Webhook response received: {webhook_response is not None}, has dynamic_variables: {webhook_response and 'dynamic_variables' in webhook_response if webhook_response else False}")
                                    
                                    # Apply dynamic variables from webhook response
                                    if webhook_response and "dynamic_variables" in webhook_response:
                                        response_vars = webhook_response["dynamic_variables"]
                                        _logger.info(f"📥 Applying {len(response_vars)} dynamic variables from webhook response")
                                        
                                        # Merge with existing dynamic variables
                                        if conn.dynamic_variables:
                                            conn.dynamic_variables.update(response_vars)
                                        else:
                                            conn.dynamic_variables = response_vars
                                        
                                        # Apply first_message if provided
                                        if "first_message" in response_vars:
                                            if conn.agent_config:
                                                conn.agent_config["first_message"] = response_vars["first_message"]
                                                _logger.info(f"✅ Applied first_message from webhook: '{response_vars['first_message'][:50]}...'")
                                            else:
                                                _logger.warning("⚠️ Cannot apply first_message - agent_config not loaded yet")
                                else:
                                    # For OUTBOUND calls: Fire-and-forget webhook
                                    asyncio.create_task(send_webhook(
                                        webhook.webhook_url,
                                        "call.started",
                                        {
                                            "conversation_id": current_call_sid,
                                            "agent_id": agent_id,
                                            "direction": call_direction,
                                            "status": "in-progress",
                                            "phone_number": caller_phone
                                        }
                                    ))
                    finally:
                        db.close()

                    # âœ… CRITICAL: Initialize resampler ONCE per connection
                    dummy_state = None
                    try:
                        _, dummy_state = audioop.ratecv(
                            b'\x00' * 3200, 2, 1, 16000, 8000, dummy_state
                        )
                        conn.resampler_state = dummy_state
                        conn.resampler_initialized = True
                        _logger.info("ðŸŽµ Resampler pre-initialized for this connection")
                    except Exception as e:
                        _logger.warning("Failed to pre-init resampler: %s", e)

                    await setup_streaming_stt(current_call_sid)
                    conn.tts_task = asyncio.create_task(
                        stream_tts_worker(current_call_sid))

                await asyncio.sleep(0.1)
                greeting = None

                # ✨ Use agent's first_message or default greeting
                if conn and conn.agent_config and conn.agent_config.get("first_message"):
                    # Replace {{variable}} placeholders in first_message
                    greeting = conn.agent_config["first_message"]
                    if conn.dynamic_variables:
                        for key, value in conn.dynamic_variables.items():
                            greeting = greeting.replace(f"{{{{{key}}}}}", str(value))
                else:
                    greeting = "hello there! this is default greeting from AI assistant. How can I help you today?"
                if conn and conn.dynamic_variables and greeting:
                    for key, value in conn.dynamic_variables.items():
                        greeting = greeting.replace(f"{{{{{key}}}}}", str(value))
                
                # 🔍 DEBUG: Verify overrides are still set before greeting
                _logger.info(f"🎯 BEFORE GREETING - conn.custom_voice_id: '{conn.custom_voice_id}'")
                _logger.info(f"🎯 BEFORE GREETING - conn.agent_config voice: '{conn.agent_config.get('voice_id') if conn.agent_config else None}'")
                
                await speak_text_streaming(current_call_sid, greeting)
                
                # ✨ CAPTURE GREETING IN TRANSCRIPT (like ElevenLabs)
                # This ensures we have a transcript even if user hangs up immediately
                if conn and greeting:
                    conn.conversation_history.append({
                        "user": "[Call Started]",
                        "assistant": greeting,
                        "timestamp": time.time()
                    })
                    _logger.info(f"✅ Greeting captured in conversation history")

            elif event == "media":
                if not current_call_sid:
                    continue

                media_data = data.get("media", {})
                payload_b64 = media_data.get("payload")

                if payload_b64:
                    try:
                        chunk = base64.b64decode(payload_b64)
                        conn = manager.get(current_call_sid)

                        if not conn:
                            continue

                        # Send to Deepgram
                        if conn.deepgram_live:
                            try:
                                conn.deepgram_live.send(chunk)
                            except Exception as e:
                                pass

                        energy = calculate_audio_energy(chunk)
                        update_baseline(conn, energy)

                        now = time.time()

                        # Calculate energy threshold
                        energy_threshold = max(
                            conn.baseline_energy * INTERRUPT_BASELINE_FACTOR,
                            INTERRUPT_MIN_ENERGY
                        )

                        # ========================================
                        # âœ… SMART VAD VALIDATION & TIMEOUT LOGIC
                        # ========================================

                        if conn.vad_triggered_time and conn.user_speech_detected:
                            time_since_vad = now - conn.vad_triggered_time

                            # Check if we're seeing actual speech energy
                            if energy >= energy_threshold:
                                # âœ… Real speech detected
                                conn.last_valid_speech_energy = energy
                                conn.energy_drop_time = None  # Reset drop timer

                                # Validate VAD after short period
                                if not conn.vad_validated and time_since_vad >= conn.vad_validation_threshold:
                                    conn.vad_validated = True
                                    _logger.info(
                                        f"âœ… VAD validated after {time_since_vad*1000:.0f}ms (energy: {energy})")

                                if not conn.speech_start_time:
                                    conn.speech_start_time = now

                            else:
                                # Low energy - but is it silence or just a pause?

                                if conn.vad_validated:
                                    # âœ… VAD was real - this is just low energy during speech (normal)
                                    # Track when energy dropped
                                    if conn.energy_drop_time is None:
                                        conn.energy_drop_time = now

                                    # Only clear VAD if energy stays low for extended period
                                    # AND we have FINAL or interim text (meaning Deepgram also thinks speech ended)
                                    low_energy_duration = now - conn.energy_drop_time

                                    if low_energy_duration >= 1.5:  # 1.5s of low energy
                                        # Check if Deepgram also stopped detecting speech
                                        time_since_last_text = now - conn.last_interim_time if conn.last_interim_time else 999

                                        if time_since_last_text > 1.0:  # No text for 1s
                                            _logger.info(
                                                f"âœ… VAD cleared naturally (low energy: {low_energy_duration:.1f}s, no text: {time_since_last_text:.1f}s)")
                                            conn.user_speech_detected = False
                                            conn.speech_start_time = None
                                            conn.vad_triggered_time = None
                                            conn.vad_validated = False
                                            conn.energy_drop_time = None
                                else:
                                    # âŒ VAD not validated yet - might be false positive
                                    # Give it 1s to validate (reduced from 3s)
                                    if time_since_vad >= 1.0:
                                        _logger.warning(
                                            f"âš ï¸ VAD timeout - false positive (duration: {time_since_vad:.1f}s)")
                                        conn.user_speech_detected = False
                                        conn.speech_start_time = None
                                        conn.vad_triggered_time = None
                                        conn.vad_validated = False
                                        conn.energy_drop_time = None

                        # ========================================
                        # âœ… INTERRUPT DETECTION (unchanged logic)
                        # ========================================

                        if conn.currently_speaking and conn.user_speech_detected and not conn.interrupt_requested:
                            # Only interrupt if VAD has been validated (real speech)
                            if conn.vad_validated and conn.speech_start_time:
                                user_speaking_duration = (
                                    now - conn.speech_start_time) * 1000.0

                                if user_speaking_duration < 500:
                                    continue

                                conn.speech_energy_buffer.append((now, energy))

                                vad_dur_ms = (
                                    now - conn.speech_start_time) * 1000.0
                                buf = list(conn.speech_energy_buffer)

                                window_ms = 300
                                cutoff_time = now - (window_ms / 1000.0)
                                recent_packets = [
                                    (t, e) for t, e in buf if t >= cutoff_time]

                                high_energy_count = sum(
                                    1 for _, e in recent_packets if e >= energy_threshold)
                                total_count = len(recent_packets)
                                energy_percentage = (
                                    high_energy_count / total_count * 100) if total_count > 0 else 0

                                peak_energy = max(
                                    (e for _, e in recent_packets), default=0)

                                time_since_last_interrupt = now - conn.last_interrupt_time
                                debounced = time_since_last_interrupt >= (
                                    INTERRUPT_DEBOUNCE_MS / 1000.0)

                                vad_ok = vad_dur_ms >= INTERRUPT_MIN_SPEECH_MS
                                energy_ok = energy_percentage >= 60 or peak_energy >= (
                                    conn.baseline_energy * INTERRUPT_BASELINE_FACTOR)
                                current_energy_ok = energy >= (
                                    energy_threshold * 0.8)

                                all_checks_pass = vad_ok and energy_ok and current_energy_ok and debounced

                                if all_checks_pass:
                                    conn.interrupt_requested = True
                                    conn.last_interrupt_time = now
                                    _logger.info(
                                        "ðŸ›‘ INTERRUPT! VAD: %.0fms | Energy: %.0f%% | Peak: %d | Threshold: %d",
                                        vad_dur_ms, energy_percentage, peak_energy, energy_threshold
                                    )

                        # ========================================
                        # âœ… PROCESS TRANSCRIPT
                        # ========================================

                        if not conn.currently_speaking and not conn.interrupt_requested:
                            if processing_task is None or processing_task.done():
                                processing_task = asyncio.create_task(
                                    process_streaming_transcript(
                                        current_call_sid)
                                )

                    except Exception as e:
                        pass

            elif event == "stop":
                break

    except WebSocketDisconnect:
        _logger.info(f"📞 WebSocket disconnected for call: {current_call_sid}")
    except Exception as e:
        _logger.error(f"❌ WebSocket error: {e}")
    finally:
        try:
            if processing_task and not processing_task.done():
                processing_task.cancel()
        except:
            pass

        try:
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
        except:
            pass

        # ✅ CRITICAL: Save transcript BEFORE disconnecting (in case /voice/status comes later)
        if current_call_sid:
            conn = manager.get(current_call_sid)
            if conn:
                _logger.info(f"💾 Saving transcript on WebSocket disconnect for: {current_call_sid}")
                _logger.info(f"   - conversation_history entries: {len(conn.conversation_history)}")
                await save_conversation_transcript(current_call_sid, conn)
            else:
                _logger.warning(f"⚠️ No connection found on WebSocket disconnect for: {current_call_sid}")
            
            try:
                await manager.disconnect(current_call_sid)
            except:
                pass

        try:
            await websocket.close()
        except:
            pass


@app.api_route("/", methods=["GET", "POST"])
async def index_page():
    return {
        "status": "ok",
        "message": "Twilio RAG Voice System - GPU + SMART VOICE INTERRUPTS + TRANSFER CONFIRMATION",
        "device": str(DEVICE),
        "features": [
            "âœ… Transfer requires user confirmation",
            "âœ… End call is immediate (no confirmation)",
            "âœ… Interrupts on real voice (configurable)",
            "âœ… GPU-accelerated RAG",
            "âœ… Streaming STT/TTS pipeline",
            "âœ… Smart conversation handling",
            f"âœ… {SILENCE_THRESHOLD_SEC}s silence before processing"
        ]
    }


@app.get("/gpu-status")
async def gpu_status():
    """🚀 GPU status"""
    status = {
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        try:
            status.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "memory": {
                    "total_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}",
                    "allocated_gb": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}",
                    "free_gb": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f}"
                },
            })
        except Exception as e:
            status["gpu_error"] = str(e)

    try:
        result = ollama.list()
        status["ollama"] = {
            "models": [m["name"] for m in result.get("models", [])],
            "current_model": OLLAMA_MODEL
        }
    except Exception as e:
        status["ollama"] = {"error": str(e)}

    status["embedding"] = {
        "model": EMBED_MODEL,
        "device": str(embedder.device),
    }

    return status


@app.post("/voice/outbound")
@app.get("/voice/outbound")
async def voice_outbound(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    
    # Check if recording is enabled for this call
    call_data = pending_call_data.get(call_sid, {})
    enable_recording = call_data.get("enable_recording", False)
    
    response = VoiceResponse()
    
    # Enable recording if requested
    if enable_recording:
        response.record(
            recording_status_callback=f"{PUBLIC_URL}/recording-callback",
            recording_status_callback_method="POST",
            recording_status_callback_event="completed"
        )
        _logger.info(f"🎙️ Recording enabled for call: {call_sid}")
    
    response.say(" ")
    response.pause(length=0)

    connect = Connect()
    connect.stream(url=f"wss://{public_ws_host()}/media-stream")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


@app.post("/voice/inbound")
@app.get("/voice/inbound")
async def voice_inbound(request: Request):
    """
    Handle incoming calls - route to appropriate agent
    
    ✨ ELEVENLABS-COMPATIBLE: Always creates conversation record and tracks status
    """
    form = await request.form()
    from_number = form.get("From", "")
    to_number = form.get("To", "")
    call_sid = form.get("CallSid", "")
    
    _logger.info(f"📞 Inbound call: from={from_number}, to={to_number}, call_sid={call_sid}")
    
    db = SessionLocal()
    try:
        # Try to find agent linked to this phone number
        phone_record = db.query(PhoneNumber).filter(
            PhoneNumber.phone_number == to_number,
            PhoneNumber.is_active == True
        ).first()
        
        agent = None
        if phone_record and phone_record.agent_id:
            agent = db.query(Agent).filter(
                Agent.agent_id == phone_record.agent_id,
                Agent.is_active == True
            ).first()
        
        # Fallback to first active agent if no phone number mapping
        if not agent:
            agent = db.query(Agent).filter(Agent.is_active == True).first()
        
        # ✨ ALWAYS store call data for WebSocket (like ElevenLabs)
        pending_call_data[call_sid] = {
            "agent_id": agent.agent_id if agent else None,
            "dynamic_variables": {"caller_number": from_number},
            "custom_voice_id": None,
            "custom_model": None,
            "custom_first_message": None,
            "from_number": from_number,  # ✅ Caller's phone number
            "to_number": to_number,      # Agent's phone number
            "enable_recording": False,
            "direction": "inbound"
        }
        
        # ✨ ALWAYS create conversation record (like ElevenLabs)
        conversation = Conversation(
            conversation_id=call_sid,
            agent_id=agent.agent_id if agent else None,
            phone_number=from_number,
            status="initiated",
            dynamic_variables={"caller_number": from_number, "direction": "inbound"},
            call_metadata={"direction": "inbound", "to_number": to_number}
        )
        db.add(conversation)
        db.commit()
        
        if agent:
            _logger.info(f"✅ Inbound call routed to agent: {agent.agent_id} ({agent.name})")
        else:
            _logger.warning("⚠️ No active agent found for inbound call - using default behavior")
        
        # ✨ ALWAYS send webhooks (like ElevenLabs)
        webhooks = db.query(WebhookConfig).filter(
            WebhookConfig.is_active == True
        ).all()
        
        # Filter webhooks by agent_id or global (agent_id == None)
        for webhook in webhooks:
            should_send = False
            if webhook.agent_id is None:
                should_send = True  # Global webhook
            elif agent and webhook.agent_id == agent.agent_id:
                should_send = True  # Agent-specific webhook
            
            if should_send and ("call.initiated" in webhook.events or not webhook.events):
                asyncio.create_task(send_webhook(
                    webhook.webhook_url,
                    "call.initiated",
                    {
                        "conversation_id": call_sid,
                        "agent_id": agent.agent_id if agent else None,
                        "from_number": from_number,
                        "to_number": to_number,
                        "direction": "inbound",
                        "status": "initiated"
                    }
                ))
    finally:
        db.close()
    
    response = VoiceResponse()
    response.say(" ")
    response.pause(length=0)
    
    connect = Connect()
    connect.stream(url=f"wss://{public_ws_host()}/media-stream")
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/make-call")
async def make_call(request: CallRequest):
    try:
        to_number = request.to_number

        _logger.info(f"📞 Starting outbound call: to={to_number}")  
        webhook = f"{PUBLIC_URL.rstrip('/')}/voice/outbound"
        status_callback_url = f"{PUBLIC_URL.rstrip('/')}/voice/status"

        call_sid = twilio_client.calls.create(
            to=to_number,
            from_=+15108963495,  # your fixed Twilio number
            url=webhook,
            method="POST",
            status_callback=status_callback_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )

        return {
            "success": True,
            "message": "Call initiated successfully",
            "call_sid": call_sid.sid,
            "to": to_number
        }

    except Exception as e:
        _logger.exception("âŒ Call initiation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/status")
async def voice_status(request: Request):
    """Enhanced voice status handler with transcript saving and webhooks"""
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    
    _logger.info(f"📞 Call status update: {call_sid} -> {call_status}")
    
    if call_status in {"completed", "failed", "busy", "no-answer", "canceled"} and call_sid:
        conn = manager.get(call_sid)
        
        # Save transcript before disconnecting
        if conn:
            await save_conversation_transcript(call_sid, conn)
        
        # Handle call end (update DB and send webhooks)
        await handle_call_end(call_sid, call_status)
        
        # Disconnect
        await manager.disconnect(call_sid)
        
        # Clean up pending call data
        if call_sid in pending_call_data:
            del pending_call_data[call_sid]

    return PlainTextResponse("OK")


@app.get("/health")
async def health():
    """Health check"""
    health_data = {
        "status": "ok",
        "mode": "GPU + FIXED_INTERRUPTS + CONFIRMATION + 1s_SILENCE",
        "device": str(DEVICE),
        "docs_count": collection.count(),
        "active_connections": len(manager._conns),
        "confirmation_enabled": True,
        "silence_threshold_sec": SILENCE_THRESHOLD_SEC,
        "interrupt_settings": {
            "enabled": INTERRUPT_ENABLED,
            "min_energy": INTERRUPT_MIN_ENERGY,
            "baseline_factor": INTERRUPT_BASELINE_FACTOR,
            "min_speech_ms": INTERRUPT_MIN_SPEECH_MS,
            "require_text": INTERRUPT_REQUIRE_TEXT
        }
    }

    if torch.cuda.is_available():
        health_data["gpu_memory_allocated_gb"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}"

    return health_data


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 50) -> List[str]:
    """
    Advanced semantic chunking with overlap for context preservation
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for i, sentence in enumerate(sentences):
        # If adding this sentence would exceed chunk size
        if current_chunk and len(current_chunk) + len(sentence) + 1 > size:
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap from previous chunk
            if overlap > 0:
                # Take last few sentences from current chunk for overlap
                prev_sentences = current_chunk.split('. ')
                overlap_sentences = prev_sentences[-2:] if len(
                    prev_sentences) > 2 else prev_sentences[-1:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter and deduplicate
    final_chunks = []
    seen = set()
    for chunk in chunks:
        if len(chunk) < 25:
            continue
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:12]
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            final_chunks.append(chunk)

    _logger.info(
        f"ðŸ“ Created {len(final_chunks)} overlapping chunks (size: {size}, overlap: {overlap})")
    return final_chunks


def build_index_from_file(path: str = DATA_FILE) -> Tuple[int, int]:
    """Build ChromaDB index using contextual chunking"""

    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_FILE not found: {path}")

    _logger.info(f"ðŸ“– Reading data from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        _logger.warning("âš  DATA_FILE is empty.")
        return (0, 0)

    # Use the new chunking with overlap
    # 50 character overlap
    docs = _chunk_text(raw_text, CHUNK_SIZE, overlap=50)

    # Clear existing collection
    try:
        chroma_client.delete_collection("docs")
    except:
        pass
    collection = chroma_client.get_or_create_collection("docs")

    metadatas = []
    ids = []

    for i, doc in enumerate(docs):
        metadatas.append({
            "chunk_id": i,
            "length": len(doc),
            "word_count": len(doc.split()),
            "type": "contextual_chunk"
        })
        ids.append(f"ctx_{i}_{uuid.uuid4().hex[:8]}")

    total = len(docs)
    if total == 0:
        _logger.warning("âš  No valid chunks found.")
        return (0, 0)

    _logger.info(f"ðŸ”„ Generating {total} embeddings...")

    start = time.time()

    with torch.no_grad():
        embeddings = embedder.encode(
            docs,
            device=DEVICE,
            batch_size=64 if DEVICE == "cuda" else 32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        ).tolist()

    duration = time.time() - start
    _logger.info(f"âœ… Embeddings done in {duration:.2f}s")

    # Batch insertion
    CHROMA_MAX_BATCH = 5000

    for start_idx in range(0, total, CHROMA_MAX_BATCH):
        end_idx = start_idx + CHROMA_MAX_BATCH
        collection.add(
            documents=docs[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=ids[start_idx:end_idx]
        )

    _logger.info(f"âœ… Contextual index built: {total} meaningful chunks")
    return (total, total)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "build", "test"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9001, type=int)
    args = parser.parse_args()

    if args.mode == "build":
        docs, chunks = build_index_from_file(DATA_FILE)
        print(f"âœ… Built index: {docs} docs, {chunks} chunks")
        print(f"🚀 Device used: {DEVICE}")
    elif args.mode == "test":
        print(f"Test mode (device: {DEVICE}):")

        async def test_query():
            while True:
                q = input("> ").strip()
                if q.lower() in {"exit", "quit"}:
                    break
                result = ""
                print("Response: ", end="", flush=True)
                async for token in query_rag_streaming(q):
                    result += token
                    print(token, end="", flush=True)
                print("\n")

        asyncio.run(test_query())
    else:
        _logger.info("🚀 Starting server on %s:%s", args.host, args.port)
        _logger.info(f"🔥 GPU: {DEVICE}")
        _logger.info("âœ… Transfer confirmation: ENABLED")
        _logger.info(
            f"â±ï¸  Silence threshold: {SILENCE_THRESHOLD_SEC}s (utterance_end={UTTERANCE_END_MS}ms)")
        _logger.info(
            f"🎯 Interrupt: enabled={INTERRUPT_ENABLED}, min_speech={INTERRUPT_MIN_SPEECH_MS}ms, "
            f"min_energy={INTERRUPT_MIN_ENERGY}, baseline_factor={INTERRUPT_BASELINE_FACTOR}, "
            f"require_text={INTERRUPT_REQUIRE_TEXT}"
        )
        uvicorn.run("agent_service:app",
                    host=args.host,
                    port=args.port,
                    reload=False,
                    timeout_keep_alive=60,
                    timeout_graceful_shutdown=30,
                    ws_ping_interval=10.0,    # Add WebSocket ping
                    ws_ping_timeout=10.0
                    )

