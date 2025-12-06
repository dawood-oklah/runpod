# MedGemma RunPod Dockerfile
# Optimized for A6000 GPU (48GB VRAM) with 8-bit quantization
#
# Source: RunPod documentation
# URL: https://www.runpod.io/articles/guides/deploy-fastapi-applications-gpu-cloud
# Verified: 2025-12-06

# Base image with CUDA support for PyTorch
# Using NVIDIA PyTorch container for optimal GPU performance
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # HuggingFace cache directory (for model caching)
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Disable tokenizers parallelism warning
    TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: bitsandbytes requires CUDA and will be compiled during install
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env.example ./.env.example

# Create cache directory for HuggingFace models
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache

# Create non-root user for security (optional, RunPod typically runs as root)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# Expose port for FastAPI
# RunPod requires 0.0.0.0 binding
EXPOSE 8000

# Health check for RunPod
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
# Using uvicorn with settings optimized for GPU workloads
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
