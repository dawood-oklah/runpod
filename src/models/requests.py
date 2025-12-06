"""
Pydantic request models for MedGemma API.

These models define the structure for all API requests,
with validation for medical AI use cases.

Source: OpenAI API specification for compatibility
URL: https://platform.openai.com/docs/api-reference/chat
Verified: 2025-12-06
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """
    A single message in a conversation.

    Follows OpenAI chat format for compatibility.
    """

    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message author"
    )
    content: str = Field(
        description="The content of the message",
        min_length=1,
        max_length=100000,  # ~25K tokens approximate
    )


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Supports multi-turn conversations with system prompts.
    For image analysis, use the dedicated /v1/medical/analyze-image endpoint.
    """

    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation",
        min_length=1,
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Maximum tokens to generate (default: 2048, max: 8192)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0, default: 0.7)",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0.0-1.0, default: 0.9)",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Top-k sampling parameter (1-100, default: 50)",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream responses (not yet supported)",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure at least one user message exists."""
        has_user = any(msg.role == "user" for msg in v)
        if not has_user:
            raise ValueError("At least one user message is required")
        return v


class MedicalQueryRequest(BaseModel):
    """
    Request for text-only medical queries.

    Optimized for long medical prompts (up to 10K tokens).
    Examples:
    - Clinical case analysis
    - Differential diagnosis
    - Treatment recommendations
    - Medical literature questions
    """

    query: str = Field(
        description="Medical question or clinical case description",
        min_length=1,
        max_length=100000,
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=10000,
        description="Custom system prompt for specialized medical context",
    )
    conversation_history: list[ChatMessage] | None = Field(
        default=None,
        description="Previous conversation messages for context",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Maximum tokens to generate",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    include_disclaimer: bool = Field(
        default=True,
        description="Include medical disclaimer in response",
    )


class ImageAnalysisRequest(BaseModel):
    """
    Request for medical image analysis.

    Supports multiple modalities:
    - Radiology (X-ray, CT, MRI)
    - Pathology
    - Dermatology
    - Ophthalmology (retinal scans)
    - Clinical photography

    Images should be base64 encoded or provided as data URLs.
    """

    images: list[str] = Field(
        description="List of base64-encoded images or data URLs",
        min_length=1,
        max_length=4,
    )
    query: str = Field(
        description="Question or instruction about the image(s)",
        min_length=1,
        max_length=50000,
    )
    image_modality: str | None = Field(
        default=None,
        description="Type of medical image (e.g., 'chest_xray', 'ct_scan', 'mri', 'pathology', 'dermatology')",
    )
    clinical_context: str | None = Field(
        default=None,
        max_length=10000,
        description="Additional clinical context (patient history, symptoms, etc.)",
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=10000,
        description="Custom system prompt",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description="Maximum tokens to generate",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    structured_output: bool = Field(
        default=False,
        description="Request structured findings format",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str]) -> list[str]:
        """Validate image data format."""
        for i, img in enumerate(v):
            if not img:
                raise ValueError(f"Image at index {i} is empty")
            # Check for valid base64 or data URL
            if not (
                img.startswith("data:image/")
                or cls._is_valid_base64(img)
            ):
                raise ValueError(
                    f"Image at index {i} must be base64-encoded or a data URL"
                )
        return v

    @staticmethod
    def _is_valid_base64(s: str) -> bool:
        """Check if string is valid base64."""
        import base64
        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False


class MedicalReportRequest(BaseModel):
    """
    Request for generating structured medical reports from images.

    Generates comprehensive reports including:
    - Findings
    - Impressions
    - Recommendations
    - Differential diagnoses (if applicable)
    """

    images: list[str] = Field(
        description="List of base64-encoded images",
        min_length=1,
        max_length=4,
    )
    report_type: Literal[
        "radiology",
        "pathology",
        "dermatology",
        "ophthalmology",
        "general",
    ] = Field(
        default="general",
        description="Type of medical report to generate",
    )
    patient_info: str | None = Field(
        default=None,
        max_length=5000,
        description="Relevant patient information (age, sex, history)",
    )
    clinical_indication: str | None = Field(
        default=None,
        max_length=5000,
        description="Reason for examination",
    )
    comparison_studies: str | None = Field(
        default=None,
        max_length=2000,
        description="Previous studies available for comparison",
    )
    max_tokens: int | None = Field(
        default=4096,
        ge=1,
        le=8192,
        description="Maximum tokens for report",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str]) -> list[str]:
        """Validate image data."""
        # Reuse validation from ImageAnalysisRequest
        return ImageAnalysisRequest.validate_images(v)
