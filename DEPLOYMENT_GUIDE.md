# MedGemma RunPod Deployment Guide

Complete step-by-step guide to deploy MedGemma-27B on RunPod with A6000 GPU.

**Estimated Time:** 30-45 minutes (first deployment)

---

## Table of Contents

1. [Prerequisites](#step-1-prerequisites)
2. [HuggingFace Setup](#step-2-huggingface-setup)
3. [Prepare Your Code](#step-3-prepare-your-code)
4. [Build Docker Image](#step-4-build-docker-image)
5. [Push to Docker Hub](#step-5-push-to-docker-hub)
6. [Create RunPod Account](#step-6-create-runpod-account)
7. [Deploy Pod](#step-7-deploy-pod)
8. [Configure Environment](#step-8-configure-environment)
9. [Verify Deployment](#step-9-verify-deployment)
10. [Test the API](#step-10-test-the-api)
11. [Troubleshooting](#troubleshooting)

---

## Step 1: Prerequisites

### Required Software

Install these on your local machine:

**Windows:**
```powershell
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Install Git (if not already installed)
winget install Git.Git

# Install Python 3.10+ (for testing)
winget install Python.Python.3.11
```

**Mac:**
```bash
# Install Docker Desktop
brew install --cask docker

# Install Python
brew install python@3.11
```

**Linux:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Python
sudo apt install python3.11 python3-pip
```

### Verify Docker is Running

```bash
docker --version
# Expected: Docker version 24.x.x or higher

docker ps
# Should show empty list (no error)
```

---

## Step 2: HuggingFace Setup

### 2.1 Create HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co)
2. Click **Sign Up** (or log in if you have an account)
3. Verify your email

### 2.2 Accept MedGemma License

**IMPORTANT:** You must accept the model license before you can download it.

1. Go to [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it)
2. You'll see a gated model notice
3. Click **"Agree and access repository"**
4. Fill in the required information (research purpose, etc.)
5. Submit and wait for approval (usually instant for research use)

![Accept License](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gated-model.png)

### 2.3 Create Access Token

1. Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Configure:
   - **Name:** `medgemma-runpod`
   - **Type:** `Read` (sufficient for downloading models)
4. Click **"Generate token"**
5. **COPY THE TOKEN NOW** - you won't see it again!

Save it somewhere safe. It looks like: `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`

---

## Step 3: Prepare Your Code

### 3.1 Navigate to Project Directory

```bash
cd f:\AI\Helpers\MedGemmaRunpod
# or wherever you cloned/created the project
```

### 3.2 Create Your .env File

```bash
# Copy the example file
copy .env.example .env
```

### 3.3 Edit .env File

Open `.env` in your editor and set:

```ini
# REQUIRED: Your HuggingFace token from Step 2.3
HF_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456

# REQUIRED: Generate a secure API key
# On Windows PowerShell:
#   [System.Guid]::NewGuid().ToString()
# On Linux/Mac:
#   openssl rand -hex 32
API_KEY=your-generated-api-key-here

# Keep these defaults for A6000
MODEL_ID=google/medgemma-27b-it
QUANTIZATION=8bit
```

**Generate a secure API key:**

Windows PowerShell:
```powershell
[System.Guid]::NewGuid().ToString()
# Example output: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

Linux/Mac:
```bash
openssl rand -hex 32
# Example output: 7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a
```

---

## Step 4: Build Docker Image

### 4.1 Open Terminal in Project Directory

```bash
cd f:\AI\Helpers\MedGemmaRunpod
```

### 4.2 Build the Image

```bash
docker build -t medgemma-api:latest .
```

**Expected output:**
```
[+] Building 120.5s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [1/6] FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
 => [2/6] RUN apt-get update && apt-get install -y ...
 => [3/6] COPY requirements.txt .
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt
 => [5/6] COPY src/ ./src/
 => [6/6] RUN mkdir -p /app/.cache/huggingface
 => exporting to image
 => => naming to docker.io/library/medgemma-api:latest
```

**Build time:** 5-15 minutes (depends on internet speed)

### 4.3 Verify Build

```bash
docker images medgemma-api
```

**Expected output:**
```
REPOSITORY     TAG       IMAGE ID       CREATED          SIZE
medgemma-api   latest    abc123def456   30 seconds ago   8.5GB
```

---

## Step 5: Push to Docker Hub

### 5.1 Create Docker Hub Account

1. Go to [hub.docker.com](https://hub.docker.com)
2. Click **Sign Up** (free account works)
3. Verify your email

### 5.2 Login to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted.

### 5.3 Tag Your Image

Replace `YOUR_DOCKERHUB_USERNAME` with your actual username:

```bash
docker tag medgemma-api:latest YOUR_DOCKERHUB_USERNAME/medgemma-api:latest
```

**Example:**
```bash
docker tag medgemma-api:latest johnsmith/medgemma-api:latest
```

### 5.4 Push to Docker Hub

```bash
docker push YOUR_DOCKERHUB_USERNAME/medgemma-api:latest
```

**Expected output:**
```
The push refers to repository [docker.io/johnsmith/medgemma-api]
5f70bf18a086: Pushed
abc123def456: Pushed
...
latest: digest: sha256:abc123... size: 3456
```

**Push time:** 10-30 minutes (image is ~8GB)

### 5.5 Verify on Docker Hub

1. Go to [hub.docker.com](https://hub.docker.com)
2. Click on your profile
3. You should see `medgemma-api` repository

---

## Step 6: Create RunPod Account

### 6.1 Sign Up

1. Go to [runpod.io](https://www.runpod.io)
2. Click **Sign Up**
3. Complete registration

### 6.2 Add Credits

1. Go to [Billing](https://www.runpod.io/console/user/billing)
2. Click **Add Credits**
3. Add at least **$10** to start (A6000 costs ~$0.79/hour)

### 6.3 Verify Identity (if required)

Some GPU types require identity verification. Complete this if prompted.

---

## Step 7: Deploy Pod

### 7.1 Go to Pods Console

1. Navigate to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **+ Deploy** button

### 7.2 Select GPU Type

**Choose: NVIDIA A6000 (48 GB)**

![Select GPU](https://docs.runpod.io/assets/images/deploy-gpu-d6c4b5e3b7e3b5f3b7e3b5f3b7e3b5f3.png)

**Configuration:**
- GPU Type: **A6000**
- GPU Count: **1**
- Select **Community Cloud** (cheaper) or **Secure Cloud** (more reliable)

Click **Deploy** on your chosen option.

### 7.3 Configure Container

On the deployment screen, set:

**Container Image:**
```
YOUR_DOCKERHUB_USERNAME/medgemma-api:latest
```

**Example:**
```
johnsmith/medgemma-api:latest
```

**Container Disk:** `50 GB` (for model cache)

**Volume Disk:** `0 GB` (optional, use if you want persistent cache)

### 7.4 Expose HTTP Ports

Click **Edit Template** or look for **Expose HTTP Ports**:

```
8000
```

This exposes your FastAPI application.

### 7.5 Set Environment Variables

Click **Environment Variables** and add:

| Variable | Value |
|----------|-------|
| `HF_TOKEN` | `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456` |
| `API_KEY` | `your-generated-api-key` |
| `QUANTIZATION` | `8bit` |
| `LOG_LEVEL` | `INFO` |

**Screenshot example:**
```
┌─────────────────┬─────────────────────────────────────┐
│ Variable Name   │ Value                               │
├─────────────────┼─────────────────────────────────────┤
│ HF_TOKEN        │ hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456 │
│ API_KEY         │ my-super-secret-key-12345           │
│ QUANTIZATION    │ 8bit                                │
│ LOG_LEVEL       │ INFO                                │
└─────────────────┴─────────────────────────────────────┘
```

### 7.6 Deploy

1. Review your configuration
2. Click **Deploy** or **Continue**
3. Wait for pod to start

---

## Step 8: Configure Environment

### 8.1 Wait for Pod to Start

The pod will go through these stages:
1. **Creating** - Setting up the VM
2. **Pulling** - Downloading your Docker image
3. **Running** - Container is starting

**Expected time:** 5-10 minutes

### 8.2 Access Pod Logs

1. Click on your pod name
2. Click **Logs** tab
3. Watch for startup messages:

```
==========================================
MedGemma API - Starting Up
==========================================
GPU Information:
name, memory.total, memory.free
NVIDIA A6000, 48685 MiB, 48000 MiB

Loading MedGemma model: google/medgemma-27b-it
Quantization: 8bit
Loading processor...
Loading model (this may take several minutes)...
```

### 8.3 Wait for Model to Load

**First startup takes 10-15 minutes** because:
1. Model files download from HuggingFace (~50GB)
2. Model loads into GPU memory with quantization

You'll see this when ready:
```
MedGemma model loaded successfully!
GPU Memory - Allocated: 28.50GB, Reserved: 30.00GB
Starting FastAPI server...
API will be available at http://0.0.0.0:8000
```

---

## Step 9: Verify Deployment

### 9.1 Get Your Pod URL

1. Go to your pod in RunPod console
2. Look for **Connect** section
3. Find the HTTP port 8000 URL

It looks like:
```
https://abc123xyz-8000.proxy.runpod.net
```

### 9.2 Test Health Endpoint

Open in browser or use curl:

```bash
curl https://abc123xyz-8000.proxy.runpod.net/health
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
    "memory_allocated_gb": 28.5,
    "memory_reserved_gb": 30.0
  },
  "version": "1.0.0"
}
```

### 9.3 View API Documentation

Open in browser:
```
https://abc123xyz-8000.proxy.runpod.net/docs
```

You'll see the interactive Swagger UI with all endpoints.

---

## Step 10: Test the API

### 10.1 Test Medical Query (cURL)

```bash
curl -X POST "https://abc123xyz-8000.proxy.runpod.net/v1/medical/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What are the common symptoms of pneumonia?",
    "max_tokens": 512
  }'
```

**Expected response:**
```json
{
  "response": "Pneumonia typically presents with the following symptoms:\n\n1. **Respiratory symptoms:**\n   - Cough (may produce mucus)\n   - Shortness of breath\n   - Chest pain when breathing or coughing\n\n2. **Systemic symptoms:**\n   - Fever and chills\n   - Fatigue\n   - Sweating\n...",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 256,
    "total_tokens": 301
  },
  "model": "google/medgemma-27b-it",
  "disclaimer": "**DISCLAIMER**: This response is generated by an AI model..."
}
```

### 10.2 Test with Python

```python
import httpx

API_URL = "https://abc123xyz-8000.proxy.runpod.net"
API_KEY = "your-api-key-here"

# Test medical query
response = httpx.post(
    f"{API_URL}/v1/medical/query",
    headers={"X-API-Key": API_KEY},
    json={
        "query": "Explain the pathophysiology of type 2 diabetes mellitus.",
        "max_tokens": 1024,
    },
    timeout=120.0,
)

print(response.json()["response"])
```

### 10.3 Test OpenAI-Compatible Endpoint

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://abc123xyz-8000.proxy.runpod.net/v1",
    api_key="your-api-key-here",
)

response = client.chat.completions.create(
    model="medgemma",  # Model name is ignored, but required
    messages=[
        {"role": "system", "content": "You are a medical expert."},
        {"role": "user", "content": "What causes hypertension?"},
    ],
    max_tokens=512,
)

print(response.choices[0].message.content)
```

### 10.4 Test Image Analysis

```python
import base64
import httpx

# Read and encode an X-ray image
with open("chest_xray.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = httpx.post(
    f"{API_URL}/v1/medical/analyze",
    headers={"X-API-Key": API_KEY},
    json={
        "images": [image_b64],
        "query": "Analyze this chest X-ray and identify any abnormalities.",
        "image_modality": "chest_xray",
        "structured_output": True,
    },
    timeout=180.0,
)

print(response.json()["analysis"])
```

### 10.5 Run Full Test Suite

```bash
# Install test dependencies
pip install httpx

# Run tests
python scripts/test_api.py \
  --url https://abc123xyz-8000.proxy.runpod.net \
  --api-key your-api-key-here
```

---

## Troubleshooting

### Problem: "Model not loading"

**Symptoms:** Health check shows `"model_loaded": false`

**Solutions:**
1. Check logs for errors: `HF_TOKEN` might be invalid
2. Verify you accepted the model license on HuggingFace
3. Wait longer (first load can take 15+ minutes)

**Check logs:**
```
# In RunPod console, click Logs tab
# Look for errors like:
# - "401 Unauthorized" → Invalid HF_TOKEN
# - "403 Forbidden" → Need to accept license
# - "CUDA out of memory" → GPU too small
```

### Problem: "401 Unauthorized" when calling API

**Symptoms:** API returns authentication error

**Solutions:**
1. Verify API key matches what you set in environment variables
2. Check header name is exactly `X-API-Key` (case-sensitive)
3. Make sure you're including the header in requests

```bash
# Correct:
curl -H "X-API-Key: your-key" https://...

# Wrong:
curl -H "Authorization: Bearer your-key" https://...
curl -H "x-api-key: your-key" https://...  # Wrong case
```

### Problem: "Out of Memory" error

**Symptoms:** Pod crashes or shows CUDA OOM

**Solutions:**
1. Make sure you selected A6000 (48GB) not a smaller GPU
2. Reduce `MAX_INPUT_TOKENS` to `8000`
3. Use 4-bit quantization: set `QUANTIZATION=4bit`

### Problem: Slow responses

**Symptoms:** API takes 30+ seconds to respond

**Causes:**
- First request after startup is slow (model warm-up)
- Long prompts take longer to process
- Network latency to RunPod servers

**Solutions:**
1. Send a warm-up request after deployment
2. Reduce `max_tokens` in requests
3. Use shorter prompts when possible

### Problem: Pod keeps restarting

**Symptoms:** Pod status cycles between Running and Stopped

**Solutions:**
1. Check logs for crash reason
2. Increase container disk to 100GB
3. Make sure environment variables are set correctly

### Problem: Can't connect to pod

**Symptoms:** Connection refused or timeout

**Solutions:**
1. Verify pod is in "Running" state
2. Check you're using the correct URL (port 8000)
3. Wait 2-3 minutes after pod starts
4. Check if health endpoint works: `/health`

---

## Cost Optimization Tips

### 1. Stop Pod When Not in Use

```
# In RunPod console, click Stop on your pod
# You're only charged for storage when stopped (~$0.10/day)
```

### 2. Use Spot Instances

Select "Community Cloud" for lower prices (but may be preempted)

### 3. Cache Models with Volumes

Add a network volume to persist the model cache:
1. Create a 100GB network volume
2. Mount it at `/app/.cache/huggingface`
3. First load downloads once, restarts use cache

### 4. Monitor Usage

Check [Billing Dashboard](https://www.runpod.io/console/user/billing) regularly

---

## Quick Reference

### Your Deployment Info

Fill in after deployment:

| Item | Value |
|------|-------|
| Pod URL | `https://___-8000.proxy.runpod.net` |
| API Key | `_________________________` |
| HF Token | `hf_______________________` |
| GPU | A6000 (48GB) |
| Cost | ~$0.79/hour |

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/docs` | GET | No | API documentation |
| `/v1/chat/completions` | POST | Yes | Chat (OpenAI format) |
| `/v1/medical/query` | POST | Yes | Medical Q&A |
| `/v1/medical/analyze` | POST | Yes | Image analysis |
| `/v1/medical/report` | POST | Yes | Report generation |

### Useful Commands

```bash
# Test health
curl https://YOUR_POD_URL/health

# Test query
curl -X POST https://YOUR_POD_URL/v1/medical/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"query": "What is diabetes?", "max_tokens": 256}'
```

---

## Next Steps

1. **Integrate with your application** using the Python examples above
2. **Set up monitoring** to track API usage and costs
3. **Consider serverless** for variable workloads (different deployment)
4. **Fine-tune** MedGemma for your specific use case

---

## Support

- **RunPod Issues:** [RunPod Discord](https://discord.gg/runpod)
- **MedGemma Questions:** [HuggingFace Discussions](https://huggingface.co/google/medgemma-27b-it/discussions)
- **This Code:** Open an issue in your repository

---

*Last updated: 2025-12-06*
