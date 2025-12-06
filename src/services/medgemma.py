"""
MedGemma Service - Core model loading and inference logic.

Handles:
- Model loading with 8-bit quantization using bitsandbytes
- Text-only and multimodal (image+text) inference
- Chat template formatting
- Memory-efficient generation for long contexts (10K+ tokens)

Source: HuggingFace MedGemma-27B-IT documentation
URL: https://huggingface.co/google/medgemma-27b-it
Verified: 2025-12-06

Key specifications:
- Context window: 128K tokens
- Max output: 8,192 tokens
- Image resolution: 896x896 (256 tokens per image)
- Requires transformers >= 4.50.0
"""

import base64
import io
import logging
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class MedGemmaService:
    """Service class for MedGemma model inference."""

    def __init__(self, settings: Settings):
        """
        Initialize the MedGemma service.

        Args:
            settings: Application settings with model configuration
        """
        self.settings = settings
        self.model = None
        self.processor = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded

    def load_model(self) -> None:
        """
        Load the MedGemma model with configured quantization.

        Uses bitsandbytes for 8-bit quantization to reduce VRAM usage
        from ~54GB (bfloat16) to ~30GB (int8).

        Source: https://huggingface.co/docs/transformers/quantization/bitsandbytes
        Verified: 2025-12-06
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return

        logger.info(f"Loading MedGemma model: {self.settings.model_id}")
        logger.info(f"Quantization: {self.settings.quantization}")

        # Configure quantization
        # Source: https://huggingface.co/docs/bitsandbytes/main/en/index
        quantization_config = None
        torch_dtype = getattr(torch, self.settings.torch_dtype)

        if self.settings.quantization == "8bit":
            # 8-bit quantization: ~30GB VRAM for 27B model
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            logger.info("Using 8-bit quantization (bitsandbytes)")

        elif self.settings.quantization == "4bit":
            # 4-bit quantization: ~15GB VRAM for 27B model
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization (bitsandbytes NF4)")

        # Load processor (handles both text and images)
        # Source: https://huggingface.co/google/medgemma-27b-it
        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.settings.model_id,
            token=self.settings.hf_token if self.settings.hf_token else None,
            trust_remote_code=True,
        )

        # Load model with quantization
        # AutoModelForImageTextToText supports multimodal inputs
        logger.info("Loading model (this may take several minutes)...")
        model_kwargs = {
            "device_map": self.settings.device_map,
            "token": self.settings.hf_token if self.settings.hf_token else None,
            "trust_remote_code": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            # Full precision loading
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.settings.model_id,
            **model_kwargs,
        )

        # Set model to evaluation mode
        self.model.eval()

        self._is_loaded = True
        logger.info("MedGemma model loaded successfully!")
        self._log_memory_usage()

    def _log_memory_usage(self) -> None:
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def process_image(self, image_data: str | bytes | Image.Image) -> Image.Image:
        """
        Process image input to PIL Image.

        Args:
            image_data: Base64 string, bytes, or PIL Image

        Returns:
            PIL Image in RGB mode

        Raises:
            ValueError: If image cannot be processed
        """
        try:
            if isinstance(image_data, Image.Image):
                image = image_data
            elif isinstance(image_data, str):
                # Handle base64 encoded image
                if image_data.startswith("data:"):
                    # Remove data URL prefix (e.g., "data:image/png;base64,")
                    image_data = image_data.split(",", 1)[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError(f"Unsupported image type: {type(image_data)}")

            # Convert to RGB if necessary (MedGemma expects RGB)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if larger than max size to save memory
            # MedGemma normalizes to 896x896 internally
            max_size = self.settings.max_image_size
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}") from e

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list[Image.Image] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build message list for chat template.

        Message format follows MedGemma chat template:
        - System message with medical context
        - User messages with text and optional images
        - Assistant responses in conversation history

        Source: https://huggingface.co/google/medgemma-27b-it
        Verified: 2025-12-06
        """
        messages = []

        # Add system message
        system_text = system_prompt or self.settings.default_system_prompt
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        })

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if isinstance(content, str):
                    messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
                elif isinstance(content, list):
                    # Already in correct format
                    messages.append({"role": role, "content": content})

        # Build current user message
        user_content = []

        # Add images first (if any)
        if images:
            for image in images:
                user_content.append({"type": "image", "image": image})

        # Add text prompt
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        return messages

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list[str | bytes | Image.Image] | None = None,
        conversation_history: list[dict] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool | None = None,
        repetition_penalty: float | None = None,
    ) -> dict:
        """
        Generate a response from MedGemma.

        Args:
            prompt: User's text prompt (supports up to ~10K tokens)
            system_prompt: Optional custom system prompt
            images: Optional list of images (base64, bytes, or PIL Image)
            conversation_history: Optional previous conversation messages
            max_new_tokens: Maximum tokens to generate (default: 2048)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 50)
            do_sample: Whether to use sampling (default: True)
            repetition_penalty: Penalty for repetition (default: 1.1)

        Returns:
            Dict with 'response', 'usage', and 'model' information

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input exceeds token limits
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Process images if provided
        processed_images = None
        if images:
            if len(images) > self.settings.max_images_per_request:
                raise ValueError(
                    f"Too many images. Maximum: {self.settings.max_images_per_request}"
                )
            processed_images = [self.process_image(img) for img in images]

        # Build messages
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            images=processed_images,
            conversation_history=conversation_history,
        )

        # Apply chat template and tokenize
        # Source: https://huggingface.co/google/medgemma-27b-it
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Check input length
        input_length = inputs["input_ids"].shape[-1]
        if input_length > self.settings.max_input_tokens:
            raise ValueError(
                f"Input too long: {input_length} tokens. "
                f"Maximum: {self.settings.max_input_tokens} tokens."
            )

        logger.info(f"Input tokens: {input_length}")

        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.settings.default_max_new_tokens,
            "do_sample": do_sample if do_sample is not None else self.settings.do_sample,
            "temperature": temperature or self.settings.temperature,
            "top_p": top_p or self.settings.top_p,
            "top_k": top_k or self.settings.top_k,
            "repetition_penalty": repetition_penalty or self.settings.repetition_penalty,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Disable sampling if temperature is 0
        if gen_kwargs["temperature"] == 0:
            gen_kwargs["do_sample"] = False

        # Generate response
        logger.info("Generating response...")
        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens (exclude input)
        generated_tokens = outputs[0][input_length:]
        output_length = len(generated_tokens)

        response_text = self.processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        logger.info(f"Output tokens: {output_length}")
        self._log_memory_usage()

        return {
            "response": response_text.strip(),
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": output_length,
                "total_tokens": input_length + output_length,
            },
            "model": self.settings.model_id,
        }

    def health_check(self) -> dict:
        """
        Perform health check on the model.

        Returns:
            Dict with health status and model information
        """
        status = {
            "status": "healthy" if self._is_loaded else "loading",
            "model_loaded": self._is_loaded,
            "model_id": self.settings.model_id,
            "quantization": self.settings.quantization,
        }

        if torch.cuda.is_available():
            status["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": round(
                    torch.cuda.memory_allocated() / 1024**3, 2
                ),
                "memory_reserved_gb": round(
                    torch.cuda.memory_reserved() / 1024**3, 2
                ),
            }
        else:
            status["gpu"] = {"available": False}

        return status


# Global service instance
_medgemma_service: Optional[MedGemmaService] = None


def get_medgemma_service() -> MedGemmaService:
    """
    Get or create the global MedGemma service instance.

    Returns:
        MedGemmaService singleton instance
    """
    global _medgemma_service
    if _medgemma_service is None:
        settings = get_settings()
        _medgemma_service = MedGemmaService(settings)
    return _medgemma_service
