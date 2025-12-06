#!/bin/bash
# MedGemma RunPod Start Script
# This script is executed when the RunPod container starts
#
# Source: RunPod documentation
# URL: https://www.runpod.io/articles/guides/deploy-fastapi-applications-gpu-cloud
# Verified: 2025-12-06

set -e

echo "=========================================="
echo "MedGemma API - Starting Up"
echo "=========================================="

# Display GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Display environment info
echo "Environment:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  Model ID: ${MODEL_ID:-google/medgemma-27b-it}"
echo "  Quantization: ${QUANTIZATION:-8bit}"
echo ""

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. You need a HuggingFace token to access MedGemma."
    echo "Set it in your RunPod environment variables."
fi

if [ -z "$API_KEY" ]; then
    echo "WARNING: API_KEY not set. API will run without authentication."
    echo "This is NOT recommended for production."
fi

# Pre-download model if not cached (optional, improves startup after first run)
if [ "${PRELOAD_MODEL:-false}" = "true" ]; then
    echo "Pre-loading model..."
    python -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
model_id = os.environ.get('MODEL_ID', 'google/medgemma-27b-it')
token = os.environ.get('HF_TOKEN')
print(f'Downloading model: {model_id}')
AutoProcessor.from_pretrained(model_id, token=token)
print('Model downloaded successfully!')
"
fi

echo ""
echo "Starting FastAPI server..."
echo "API will be available at http://0.0.0.0:8000"
echo "Documentation at http://0.0.0.0:8000/docs"
echo "=========================================="

# Start the FastAPI application
exec python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level ${LOG_LEVEL:-info}
