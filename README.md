# MedGemma RunPod API

FastAPI service for hosting Google's MedGemma-27B multimodal medical AI model on RunPod.

## Features

- **Full MedGemma-27B-IT Support**: Multimodal model with text and image understanding
- **8-bit Quantization**: Optimized for A6000 GPU (48GB VRAM) using bitsandbytes
- **Long Context**: Supports up to 128K tokens (optimized for 10K+ token prompts)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI client libraries
- **Medical Image Analysis**: Radiology, pathology, dermatology, ophthalmology
- **Structured Report Generation**: Generate formal medical reports from images
- **API Key Authentication**: Secure your deployment

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth required) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |
| `/v1/medical/query` | POST | Text-only medical queries |
| `/v1/medical/analyze` | POST | Medical image analysis |
| `/v1/medical/report` | POST | Generate medical reports |

## Requirements

- **GPU**: NVIDIA A6000 (48GB) or better
- **HuggingFace Token**: Access to gated MedGemma model
- **Docker**: For building the container

## Quick Start

### 1. Prerequisites

1. **Accept MedGemma License**: Go to [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) and accept the terms
2. **Get HuggingFace Token**: Create a token at [HuggingFace Settings](https://huggingface.co/settings/tokens)

### 2. Build Docker Image

```bash
# Clone the repository
git clone <your-repo-url>
cd MedGemmaRunpod

# Build the Docker image
docker build -t medgemma-api:latest .
```

### 3. Test Locally (if you have a GPU)

```bash
# Create .env file
cp .env.example .env
# Edit .env and add your HF_TOKEN and API_KEY

# Run locally
docker run --gpus all -p 8000:8000 --env-file .env medgemma-api:latest
```

### 4. Push to Docker Hub

```bash
# Tag for Docker Hub
docker tag medgemma-api:latest yourusername/medgemma-api:latest

# Push
docker push yourusername/medgemma-api:latest
```

## RunPod Deployment

### Step 1: Create a Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy" to create a new pod
3. Select **GPU Pod** (not Serverless for this deployment)

### Step 2: Select GPU

Choose **NVIDIA A6000** (48GB VRAM) or better:
- A6000: 48GB - Recommended for 8-bit quantization
- A100 40GB: Good option
- A100 80GB: Ideal for larger batches

### Step 3: Configure Pod

**Container Image:**
```
yourusername/medgemma-api:latest
```

**Expose Ports:**
- HTTP: 8000

**Environment Variables:**
```
HF_TOKEN=hf_your_token_here
API_KEY=your-secure-api-key
QUANTIZATION=8bit
LOG_LEVEL=INFO
```

**Container Disk:** 50GB (for model cache)

**Volume Mount:** Optional - mount a network volume for persistent model cache

### Step 4: Deploy

Click "Deploy" and wait for the pod to start. First startup takes 10-15 minutes as the model downloads.

### Step 5: Access Your API

Once running, RunPod provides a URL like:
```
https://xxx-8000.proxy.runpod.net
```

Test it:
```bash
curl https://xxx-8000.proxy.runpod.net/health
```

## Usage Examples

### Python Client

```python
import httpx

API_URL = "https://xxx-8000.proxy.runpod.net"
API_KEY = "your-api-key"

# Medical Query
response = httpx.post(
    f"{API_URL}/v1/medical/query",
    headers={"X-API-Key": API_KEY},
    json={
        "query": "What are the differential diagnoses for a patient presenting with chest pain and shortness of breath?",
        "max_tokens": 1024,
    },
    timeout=120.0,
)
print(response.json()["response"])
```

### OpenAI SDK Compatible

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://xxx-8000.proxy.runpod.net/v1",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="medgemma-27b-it",  # Ignored, but required by SDK
    messages=[
        {"role": "system", "content": "You are a medical assistant."},
        {"role": "user", "content": "Explain the pathophysiology of type 2 diabetes."},
    ],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

### Image Analysis

```python
import base64
import httpx

# Read and encode image
with open("xray.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = httpx.post(
    f"{API_URL}/v1/medical/analyze",
    headers={"X-API-Key": API_KEY},
    json={
        "images": [image_b64],
        "query": "Analyze this chest X-ray and describe any abnormalities.",
        "image_modality": "chest_xray",
        "structured_output": True,
    },
    timeout=180.0,
)
print(response.json()["analysis"])
```

### Generate Medical Report

```python
response = httpx.post(
    f"{API_URL}/v1/medical/report",
    headers={"X-API-Key": API_KEY},
    json={
        "images": [image_b64],
        "report_type": "radiology",
        "patient_info": "65-year-old male, smoker",
        "clinical_indication": "Persistent cough for 3 weeks",
    },
    timeout=180.0,
)
print(response.json()["report"])
```

### cURL Examples

```bash
# Health check
curl https://xxx-8000.proxy.runpod.net/health

# Medical query
curl -X POST https://xxx-8000.proxy.runpod.net/v1/medical/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "What causes hypertension?", "max_tokens": 512}'

# Chat completion (OpenAI format)
curl -X POST https://xxx-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain diabetes mellitus type 2"}
    ],
    "max_tokens": 512
  }'
```

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (required) | API key for authentication |
| `HF_TOKEN` | (required) | HuggingFace token for model access |
| `MODEL_ID` | `google/medgemma-27b-it` | Model to use |
| `QUANTIZATION` | `8bit` | Quantization: none, 4bit, 8bit |
| `MAX_INPUT_TOKENS` | `12000` | Max input tokens |
| `MAX_OUTPUT_TOKENS` | `4096` | Max output tokens |
| `DEFAULT_MAX_NEW_TOKENS` | `2048` | Default generation length |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Memory Usage

| Configuration | VRAM Usage | Recommended GPU |
|--------------|------------|-----------------|
| 8-bit + 10K context | ~30-35 GB | A6000 (48GB) |
| 8-bit + 20K context | ~35-40 GB | A6000 (48GB) |
| 4-bit + 10K context | ~15-18 GB | RTX 3090 (24GB) |
| Full precision | ~55+ GB | A100 80GB |

## Project Structure

```
MedGemmaRunpod/
├── src/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── auth.py           # API key authentication
│   ├── models/
│   │   ├── requests.py   # Request schemas
│   │   └── responses.py  # Response schemas
│   └── services/
│       └── medgemma.py   # Model service
├── scripts/
│   ├── start.sh          # Container startup script
│   ├── build_and_push.sh # Build Docker image
│   └── test_api.py       # API test script
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### Model Not Loading

1. Check HuggingFace token is valid
2. Verify you accepted the model license
3. Check GPU has enough VRAM
4. Check logs: `docker logs <container_id>`

### Out of Memory

1. Reduce `MAX_INPUT_TOKENS`
2. Use 4-bit quantization instead
3. Upgrade to larger GPU

### Slow Inference

1. First request is slow (model warm-up)
2. Long prompts take longer
3. Consider reducing `MAX_OUTPUT_TOKENS`

### Authentication Errors

1. Verify API key is set correctly
2. Check header name: `X-API-Key`
3. API key is case-sensitive

## Disclaimer

MedGemma is an AI model for research and development purposes. It is NOT a certified medical device and should NOT be used for clinical decision-making without proper validation and oversight by qualified healthcare professionals.

## License

This project is provided for educational and research purposes. MedGemma model usage is subject to Google's terms of service.

## References

- [MedGemma on HuggingFace](https://huggingface.co/google/medgemma-27b-it)
- [RunPod Documentation](https://www.runpod.io/articles/guides/deploy-fastapi-applications-gpu-cloud)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [bitsandbytes Quantization](https://huggingface.co/docs/bitsandbytes)
