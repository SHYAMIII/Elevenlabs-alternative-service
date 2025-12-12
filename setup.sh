#!/bin/bash

echo "🚀 Starting Twilio RAG full setup..."

### -----------------------
### Step 1: System Update
### -----------------------
apt-get update && apt-get upgrade -y

apt-get install -y \
    git wget curl vim tmux htop net-tools build-essential software-properties-common

### -----------------------
### Step 2: Python 3.10 Setup
### -----------------------
if ! python3.10 --version &>/dev/null; then
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update
    apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

### -----------------------
### Step 3: Create Project
### -----------------------
mkdir -p /workspace/twilio-rag
cd /workspace/twilio-rag

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

### -----------------------
### Step 4: Install PyTorch
### -----------------------
CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')

if [[ "$CUDA_VERSION" == 12* ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

### -----------------------
### Step 5: Install Requirements
### -----------------------
if [ -f "requirement.txt" ]; then
    pip install -r requirement.txt
else
    echo "⚠️ requirement.txt not found! Upload it."
fi

### -----------------------
### Step 6: Install OLLAMA
### -----------------------
curl -fsSL https://ollama.com/install.sh | sh

ollama serve > /var/log/ollama.log 2>&1 &

sleep 5

ollama pull mistral:7b-instruct

### -----------------------
### Step 7: Setup Data Folder
### -----------------------
mkdir -p /workspace/twilio-rag/data
echo "⚠️ Upload your data.json or data.txt into /workspace/twilio-rag/data/"
sleep 2

### -----------------------
### Step 8: Build Chroma Index
### -----------------------
if [ -f "twilio_rag.py" ]; then
    python3 twilio_rag.py build
else
    echo "⚠️ twilio_rag.py not found! Upload your file."
fi

### -----------------------
### Step 9: Install Ngrok
### -----------------------
cd /tmp
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvzf ngrok-v3-stable-linux-amd64.tgz
mv ngrok /usr/local/bin
chmod +x /usr/local/bin/ngrok

echo "➡️ Enter your NGROK_AUTH_TOKEN:"
read NGROK_TOKEN
ngrok config add-authtoken $NGROK_TOKEN

### Start ngrok
tmux new-session -d -s ngrok 'ngrok http 9001'
sleep 3

NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | head -1 | cut -d'"' -f4)

echo "🌍 NGROK Public URL: $NGROK_URL"

### -----------------------
### Step 10: Update .env
### -----------------------
if [ -f ".env" ]; then
    sed -i "s|PUBLIC_URL=.*|PUBLIC_URL=$NGROK_URL|" .env
else
    echo "PUBLIC_URL=$NGROK_URL" > .env
fi

### -----------------------
### Step 11: Start Twilio RAG Server
### -----------------------
tmux new-session -d -s server "cd /workspace/twilio-rag && source venv/bin/activate && python3 twilio_rag.py server --host 0.0.0.0 --port 9001"

echo ""
echo "🎉 DONE!"
echo "-----------------------------------"
echo "🌍 Ngrok URL: $NGROK_URL"
echo "📞 Set this URL in Twilio Webhook"
echo "💻 Logs: tmux attach -t server"
echo "-----------------------------------"
