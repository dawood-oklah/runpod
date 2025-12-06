"""
Pydantic response models for MedGemma API.

These models define the structure for all API responses,
following OpenAI API conventions for compatibility.

Source: OpenAI API specification
URL: https://platform.openai.com/docs/api-reference/chat
Verified: 2025-12-06
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token usage information for a request."""

    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")


class Choice(BaseModel):
    """A single completion choice."""

    index: int = Field(default=0, description="Index of this choice")
    message: dict = Field(description="The generated message")
    finish_reason: Literal["stop", "length", "error"] = Field(
        default="stop",
        description="Reason for completion termination",
    )


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.

    Compatible with OpenAI client libraries for easy integration.
    """

    id: str = Field(description="Unique response identifier")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp of creation",
    )
    model: str = Field(description="Model used for generation")
    choices: list[Choice] = Field(description="List of completion choices")
    usage: UsageInfo = Field(description="Token usage statistics")


class MedicalQueryResponse(BaseModel):
    """Response for text-only medical queries."""

    response: str = Field(description="Medical response text")
    usage: UsageInfo = Field(description="Token usage statistics")
    model: str = Field(description="Model used for generation")
    disclaimer: str | None = Field(
        default=None,
        description="Medical disclaimer if included",
    )
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp of creation",
    )


class ImageAnalysisResponse(BaseModel):
    """Response for medical image analysis."""

    analysis: str = Field(description="Image analysis text")
    findings: list[str] | None = Field(
        default=None,
        description="Structured findings (if structured_output requested)",
    )
    usage: UsageInfo = Field(description="Token usage statistics")
    model: str = Field(description="Model used for generation")
    image_count: int = Field(description="Number of images analyzed")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp of creation",
    )


class MedicalReportResponse(BaseModel):
    """Response for medical report generation."""

    report: str = Field(description="Full medical report text")
    report_type: str = Field(description="Type of report generated")
    sections: dict | None = Field(
        default=None,
        description="Parsed report sections (if available)",
    )
    usage: UsageInfo = Field(description="Token usage statistics")
    model: str = Field(description="Model used for generation")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp of creation",
    )


class GPUInfo(BaseModel):
    """GPU status information."""

    available: bool = Field(description="Whether GPU is available")
    device_count: int | None = Field(default=None, description="Number of GPUs")
    current_device: int | None = Field(default=None, description="Current GPU index")
    device_name: str | None = Field(default=None, description="GPU model name")
    compute_capability: str | None = Field(default=None, description="GPU compute capability")
    memory_allocated_gb: float | None = Field(
        default=None,
        description="Allocated GPU memory in GB",
    )
    memory_reserved_gb: float | None = Field(
        default=None,
        description="Reserved GPU memory in GB",
    )
    memory_total_gb: float | None = Field(
        default=None,
        description="Total GPU memory in GB",
    )
    memory_free_gb: float | None = Field(
        default=None,
        description="Free GPU memory in GB",
    )


class PerformanceInfo(BaseModel):
    """Performance optimization information."""

    attention_implementation: str = Field(
        description="Attention implementation used (flash_attention_2, sdpa, eager)"
    )
    flash_attention_available: bool | None = Field(
        default=None,
        description="Whether Flash Attention 2 is available",
    )
    cudnn_benchmark: bool | None = Field(
        default=None,
        description="Whether cuDNN benchmark mode is enabled",
    )
    tf32_enabled: bool | None = Field(
        default=None,
        description="Whether TF32 is enabled for matrix multiplications",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "loading", "error"] = Field(
        description="Service health status"
    )
    model_loaded: bool = Field(description="Whether model is loaded")
    model_id: str = Field(description="Model identifier")
    quantization: str = Field(description="Quantization method used")
    gpu: GPUInfo = Field(description="GPU status information")
    performance: PerformanceInfo | None = Field(
        default=None,
        description="Performance optimization information",
    )
    version: str = Field(default="1.0.0", description="API version")
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp",
    )


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    param: str | None = Field(default=None, description="Parameter that caused error")
    type: str = Field(default="invalid_request_error", description="Error type")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail = Field(description="Error details")
    status_code: int = Field(description="HTTP status code")
