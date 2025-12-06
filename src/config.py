"""
Configuration settings for MedGemma RunPod API.

Environment variables are loaded from .env file or system environment.
All settings are validated using Pydantic for type safety.

Source: HuggingFace MedGemma documentation
URL: https://huggingface.co/google/medgemma-27b-it
Verified: 2025-12-06
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    api_title: str = "MedGemma API"
    api_description: str = "FastAPI service for MedGemma-27B multimodal medical AI model"
    api_version: str = "1.0.0"

    # Authentication
    api_key: str = Field(
        default="",
        description="API key for authenticating requests. Required in production."
    )
    api_key_header_name: str = "X-API-Key"

    # Model Configuration
    # Source: https://huggingface.co/google/medgemma-27b-it
    # MedGemma-27B-IT is the multimodal version supporting both text and images
    model_id: str = "google/medgemma-27b-it"

    # Quantization: 8-bit reduces VRAM from ~54GB to ~30GB
    # Source: https://huggingface.co/docs/bitsandbytes
    quantization: Literal["none", "4bit", "8bit"] = "8bit"

    # Context length: MedGemma supports up to 128K tokens
    # We set max_input_tokens to 12K to accommodate 10K prompts + system prompt + buffer
    max_input_tokens: int = 12000

    # Max output tokens: MedGemma supports up to 8192 output tokens
    max_output_tokens: int = 4096
    default_max_new_tokens: int = 2048

    # Model loading settings
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"

    # HuggingFace authentication (required for gated model)
    hf_token: str = Field(
        default="",
        description="HuggingFace token for accessing gated MedGemma model"
    )

    # Server Configuration
    host: str = "0.0.0.0"  # Required for RunPod container networking
    port: int = 8000
    workers: int = 1  # Single worker for GPU model

    # Inference settings
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Image processing settings
    # Source: MedGemma normalizes images to 896x896 and encodes to 256 tokens
    max_image_size: int = 896
    max_images_per_request: int = 4
    max_upload_size_mb: int = 20

    # Logging
    log_level: str = "INFO"

    # System prompt for medical context
    default_system_prompt: str = """You are MedGemma, an expert medical AI assistant developed by Google.
You provide accurate, evidence-based medical information while being clear about limitations.
Always recommend consulting healthcare professionals for medical decisions.
When analyzing medical images, describe findings systematically and note any limitations in image quality."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
