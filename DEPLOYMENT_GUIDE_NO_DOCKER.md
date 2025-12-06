# MedGemma RunPod Deployment Guide (No Docker)

Deploy MedGemma-27B directly on RunPod using a PyTorch template - no Docker build required.

**Estimated Time:** 20-30 minutes

---

## Table of Contents

1. [Prerequisites](#step-1-prerequisites)
2. [HuggingFace Setup](#step-2-huggingface-setup)
3. [Create RunPod Account](#step-3-create-runpod-account)
4. [Deploy Pod with PyTorch Template](#step-4-deploy-pod-with-pytorch-template)
5. [Connect to Pod](#step-5-connect-to-pod)
6. [Upload Code to Pod](#step-6-upload-code-to-pod)
7. [Install Dependencies](#step-7-install-dependencies)
8. [Configure Environment](#step-8-configure-environment)
9. [Start the API Server](#step-9-start-the-api-server)
10. [Test the API](#step-10-test-the-api)
11. [Keep Server Running](#step-11-keep-server-running)
12. [Troubleshooting](#troubleshooting)

---

## Step 1: Prerequisites

### On Your Local Machine

You need:
- Web browser
- Text editor (to edit .env file)
- Terminal/Command Prompt
- Git (optional, for cloning)

### Generate an API Key

You'll need a secure API key. Generate one:

**Windows PowerShell:**
```powershell
[System.Guid]::NewGuid().ToString()
# Output: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Linux/Mac:**
```bash
openssl rand -hex 32
# Output: 7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a
```

**Save this key** - you'll need it later.

---

## Step 2: HuggingFace Setup

### 2.1 Create/Login to HuggingFace

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up or log in

### 2.2 Accept MedGemma License

**CRITICAL - Don't skip this!**

1. Go to [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it)
2. Click **"Agree and access repository"**
3. Fill in the form (select "Research" for purpose)
4. Submit - approval is usually instant

### 2.3 Create Access Token

1. Go to [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Set:
   - Name: `medgemma-runpod`
   - Type: `Read`
4. Click **"Generate token"**
5. **COPY THE TOKEN NOW** - save it securely

Token format: `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`

---

## Step 3: Create RunPod Account

### 3.1 Sign Up

1. Go to [runpod.io](https://www.runpod.io)
2. Click **Sign Up**
3. Complete registration and verify email

### 3.2 Add Credits

1. Go to [Billing](https://www.runpod.io/console/user/billing)
2. Click **Add Credits**
3. Add **$10-20** to start

**Cost estimate:** A6000 = ~$0.79/hour

---

## Step 4: Deploy Pod with PyTorch Template

### 4.1 Go to Pods

1. Navigate to [RunPod Pods Console](https://www.runpod.io/console/pods)
2. Click **"+ Deploy"**

### 4.2 Select GPU

Choose **NVIDIA A6000 (48 GB)**:

```
┌─────────────────────────────────────────────────┐
│  GPU Selection                                  │
├─────────────────────────────────────────────────┤
│  ✓ NVIDIA A6000         48 GB    $0.79/hr      │
│    NVIDIA A100 40GB     40 GB    $1.89/hr      │
│    NVIDIA A100 80GB     80 GB    $2.49/hr      │
└─────────────────────────────────────────────────┘
```

Click **Deploy** on the A6000 option.

### 4.3 Select Template

On the deployment configuration screen:

1. Look for **"Select Template"** or **"Change Template"**
2. Search for: `PyTorch`
3. Select: **"RunPod PyTorch 2.4.0"** (or latest version)

The template should show:
- CUDA 12.x
- Python 3.10+
- PyTorch pre-installed

### 4.4 Configure Pod

Set these options:

| Setting | Value |
|---------|-------|
| **Container Disk** | `100 GB` |
| **Volume Disk** | `50 GB` (optional, for model cache) |
| **Expose HTTP Ports** | `8000` |

### 4.5 Set Environment Variables

Click **"Environment Variables"** and add:

| Variable | Value |
|----------|-------|
| `HF_TOKEN` | `hf_your_token_here` |
| `API_KEY` | `your-generated-api-key` |
| `JUPYTER_PASSWORD` | `your-jupyter-password` |

### 4.6 Deploy

Click **"Continue"** → **"Deploy"**

Wait for pod status to show **"Running"** (2-5 minutes)

---

## Step 5: Connect to Pod

### 5.1 Open Web Terminal

1. Click on your running pod
2. Click **"Connect"** button
3. Select **"Start Web Terminal"** or **"Connect to Web Terminal"**

A terminal will open in your browser.

### 5.2 Verify GPU

In the terminal, run:

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A6000        On   | 00000000:00:05.0 Off |                  Off |
| 30%   35C    P8    22W / 300W |      0MiB / 48685MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 5.3 Check Python

```bash
python --version
# Expected: Python 3.10.x or 3.11.x

pip --version
# Expected: pip 23.x or higher
```

---

## Step 6: Upload Code to Pod

### Option A: Clone from GitHub (Recommended)

If your code is on GitHub:

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/MedGemmaRunpod.git
cd MedGemmaRunpod
```

### Option B: Upload via Jupyter

1. In RunPod, click **"Connect"** → **"Connect to Jupyter Lab"**
2. In Jupyter, click the **Upload** button
3. Upload your entire `MedGemmaRunpod` folder

### Option C: Copy-Paste Code Files

Create the directory structure manually:

```bash
cd /workspace
mkdir -p MedGemmaRunpod/src/models
mkdir -p MedGemmaRunpod/src/services
cd MedGemmaRunpod
```

Then create each file using `nano` or `vim`. I'll show you how below.

---

## Step 6.1: Create Files via Terminal (Option C Details)

If you're copy-pasting, create each file:

### Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
python-multipart==0.0.17
transformers>=4.50.0
accelerate>=1.2.0
tokenizers>=0.21.0
bitsandbytes>=0.45.0
Pillow>=10.4.0
httpx>=0.28.1
python-dotenv>=1.0.1
EOF
```

### Create src/__init__.py

```bash
cat > src/__init__.py << 'EOF'
__version__ = "1.0.0"
EOF
```

### Create src/config.py

```bash
cat > src/config.py << 'EOF'
import os
from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_title: str = "MedGemma API"
    api_description: str = "FastAPI service for MedGemma-27B multimodal medical AI model"
    api_version: str = "1.0.0"
    api_key: str = Field(default="")
    api_key_header_name: str = "X-API-Key"
    model_id: str = "google/medgemma-27b-it"
    quantization: Literal["none", "4bit", "8bit"] = "8bit"
    max_input_tokens: int = 12000
    max_output_tokens: int = 4096
    default_max_new_tokens: int = 2048
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    hf_token: str = Field(default="")
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_image_size: int = 896
    max_images_per_request: int = 4
    max_upload_size_mb: int = 20
    log_level: str = "INFO"
    default_system_prompt: str = """You are MedGemma, an expert medical AI assistant developed by Google.
You provide accurate, evidence-based medical information while being clear about limitations.
Always recommend consulting healthcare professionals for medical decisions."""

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
EOF
```

### Create remaining files

For the remaining files, you have two options:

**Option 1: Use the Jupyter file upload**

This is easier for multiple files. Upload:
- `src/auth.py`
- `src/main.py`
- `src/models/__init__.py`
- `src/models/requests.py`
- `src/models/responses.py`
- `src/services/__init__.py`
- `src/services/medgemma.py`

**Option 2: Download directly from your repo**

If you have the code in a GitHub gist or raw file:

```bash
# Example - replace with your actual URLs
curl -o src/auth.py https://raw.githubusercontent.com/YOU/MedGemmaRunpod/main/src/auth.py
curl -o src/main.py https://raw.githubusercontent.com/YOU/MedGemmaRunpod/main/src/main.py
# ... etc
```

---

## Step 7: Install Dependencies

### 7.1 Navigate to Project

```bash
cd /workspace/MedGemmaRunpod
```

### 7.2 Install Python Packages

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Collecting fastapi==0.115.6
Collecting uvicorn[standard]==0.32.1
Collecting transformers>=4.50.0
...
Successfully installed fastapi-0.115.6 uvicorn-0.32.1 transformers-4.50.0 ...
```

**Install time:** 3-5 minutes

### 7.3 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoProcessor; print('Transformers OK')"
python -c "import bitsandbytes; print('bitsandbytes OK')"
```

**Expected output:**
```
PyTorch: 2.4.0+cu124
CUDA available: True
Transformers OK
bitsandbytes OK
```

---

## Step 8: Configure Environment

### 8.1 Create .env File

```bash
cd /workspace/MedGemmaRunpod

cat > .env << EOF
# HuggingFace token (replace with your actual token)
HF_TOKEN=hf_your_actual_token_here

# API Key for authentication (replace with your generated key)
API_KEY=your-secure-api-key-here

# Model settings
MODEL_ID=google/medgemma-27b-it
QUANTIZATION=8bit

# Server settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Generation settings
MAX_INPUT_TOKENS=12000
DEFAULT_MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
EOF
```

### 8.2 Edit with Your Actual Values

```bash
nano .env
```

Replace:
- `hf_your_actual_token_here` → Your HuggingFace token
- `your-secure-api-key-here` → Your generated API key

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

### 8.3 Verify .env

```bash
cat .env | grep -E "^(HF_TOKEN|API_KEY)=" | head -c 50
# Should show beginning of your tokens (don't share these!)
```

---

## Step 9: Start the API Server

### 9.1 Login to HuggingFace CLI (First Time Only)

```bash
huggingface-cli login --token $HF_TOKEN
```

Or if that doesn't work:

```bash
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
```

### 9.2 Start the Server

```bash
cd /workspace/MedGemmaRunpod
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 9.3 Watch the Startup

You'll see:

```
==========================================
MedGemma API - Starting Up
==========================================
Loading MedGemma model: google/medgemma-27b-it
Quantization: 8bit
Loading processor...
Downloading model files...
```

**First startup downloads the model (~50GB) - takes 10-15 minutes!**

When ready, you'll see:

```
MedGemma model loaded successfully!
GPU Memory - Allocated: 28.50GB, Reserved: 30.00GB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Step 10: Test the API

### 10.1 Get Your Pod URL

1. Go to RunPod console
2. Click on your pod
3. Find the **Proxy URL** for port 8000

It looks like: `https://abc123xyz-8000.proxy.runpod.net`

### 10.2 Test Health Endpoint

Open a **new browser tab** and go to:

```
https://abc123xyz-8000.proxy.runpod.net/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "google/medgemma-27b-it",
  "quantization": "8bit",
  "gpu": {
    "available": true,
    "device_name": "NVIDIA A6000",
    "memory_allocated_gb": 28.5
  }
}
```

### 10.3 View API Documentation

Go to:
```
https://abc123xyz-8000.proxy.runpod.net/docs
```

You'll see interactive Swagger UI!

### 10.4 Test Medical Query

From your **local machine** terminal:

```bash
curl -X POST "https://abc123xyz-8000.proxy.runpod.net/v1/medical/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What are the symptoms of pneumonia?",
    "max_tokens": 512
  }'
```

### 10.5 Test with Python

```python
import httpx

API_URL = "https://abc123xyz-8000.proxy.runpod.net"
API_KEY = "your-api-key-here"

response = httpx.post(
    f"{API_URL}/v1/medical/query",
    headers={"X-API-Key": API_KEY},
    json={
        "query": "Explain the difference between Type 1 and Type 2 diabetes.",
        "max_tokens": 1024,
    },
    timeout=120.0,
)

print(response.json()["response"])
```

---

## Step 11: Keep Server Running

The server stops when you close the terminal. To keep it running:

### Option A: Use Screen (Recommended)

```bash
# Install screen if not available
apt-get update && apt-get install -y screen

# Create a new screen session
screen -S medgemma

# Start the server
cd /workspace/MedGemmaRunpod
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Detach from screen: Press Ctrl+A, then D
# Server keeps running!

# To reattach later:
screen -r medgemma
```

### Option B: Use nohup

```bash
cd /workspace/MedGemmaRunpod
nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Check if running
ps aux | grep uvicorn

# View logs
tail -f server.log
```

### Option C: Use tmux

```bash
# Install tmux
apt-get update && apt-get install -y tmux

# Create session
tmux new -s medgemma

# Start server
cd /workspace/MedGemmaRunpod
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+B, then D

# Reattach:
tmux attach -t medgemma
```

---

## Step 12: Auto-Start on Pod Restart (Optional)

Create a startup script that runs when the pod starts:

```bash
cat > /workspace/start_medgemma.sh << 'EOF'
#!/bin/bash
cd /workspace/MedGemmaRunpod
source .env
export HF_TOKEN API_KEY MODEL_ID QUANTIZATION
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
EOF

chmod +x /workspace/start_medgemma.sh
```

Then in RunPod pod settings, set **Start Command** to:
```
/workspace/start_medgemma.sh
```

---

## Troubleshooting

### Problem: "Module not found" errors

```bash
# Make sure you're in the right directory
cd /workspace/MedGemmaRunpod

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
export PYTHONPATH=/workspace/MedGemmaRunpod:$PYTHONPATH
```

### Problem: "401 Unauthorized" from HuggingFace

```bash
# Re-login to HuggingFace
huggingface-cli login

# Or set token directly
export HF_TOKEN="hf_your_token_here"
```

### Problem: "CUDA out of memory"

```bash
# Check current GPU memory
nvidia-smi

# Use 4-bit quantization instead
# Edit .env:
nano .env
# Change: QUANTIZATION=4bit
```

### Problem: Server won't start

```bash
# Check for port conflicts
lsof -i :8000

# Kill any existing process
pkill -f uvicorn

# Try starting again
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Problem: "Connection refused" from browser

1. Make sure server shows "Uvicorn running on http://0.0.0.0:8000"
2. Check you're using the correct RunPod proxy URL
3. Wait 30 seconds after server starts
4. Make sure port 8000 is exposed in pod settings

### Problem: Files missing after pod restart

Pods have two storage areas:
- `/workspace` - **Persists** across restarts (if volume attached)
- Everything else - **Deleted** on restart

**Solution:** Always keep your code in `/workspace`

---

## Quick Command Reference

```bash
# Navigate to project
cd /workspace/MedGemmaRunpod

# Start server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Start in background with screen
screen -S medgemma
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
# Ctrl+A, D to detach

# Reattach to screen
screen -r medgemma

# View GPU usage
nvidia-smi

# Check server logs (if using nohup)
tail -f server.log

# Kill server
pkill -f uvicorn
```

---

## Cost Summary

| Resource | Cost |
|----------|------|
| A6000 GPU | ~$0.79/hour |
| Storage (100GB) | ~$0.10/day |
| **Total (running)** | ~$0.80/hour |
| **Total (stopped)** | ~$0.10/day |

**Tip:** Stop your pod when not using it to save money!

---

## Your Deployment Checklist

- [ ] HuggingFace account created
- [ ] MedGemma license accepted
- [ ] HuggingFace token generated: `hf_____________`
- [ ] API key generated: `_______________`
- [ ] RunPod account with credits
- [ ] Pod deployed with A6000 GPU
- [ ] Code uploaded to `/workspace/MedGemmaRunpod`
- [ ] Dependencies installed
- [ ] `.env` file configured
- [ ] Server started and running
- [ ] Health check working
- [ ] API test successful

---

## Your Deployment Info

Fill in after deployment:

| Item | Value |
|------|-------|
| Pod URL | `https://___-8000.proxy.runpod.net` |
| API Key | `_________________________` |
| HF Token | `hf_______________________` |
| GPU | A6000 (48GB) |

---

*Last updated: 2025-12-06*
