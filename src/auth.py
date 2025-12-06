"""
API Key Authentication for MedGemma API.

Implements secure API key validation using FastAPI dependency injection.
Supports both header-based and query parameter authentication.

Security considerations:
- Uses constant-time comparison to prevent timing attacks
- Logs authentication failures for security monitoring
- Returns generic error messages to prevent enumeration
"""

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class APIKeyAuth:
    """API Key authentication handler."""

    def __init__(self, settings: Settings):
        """
        Initialize API key authentication.

        Args:
            settings: Application settings containing API key
        """
        self.settings = settings
        self.api_key = settings.api_key

        # Define security schemes
        self.api_key_header = APIKeyHeader(
            name=settings.api_key_header_name,
            auto_error=False,
            description="API key passed in request header",
        )
        self.api_key_query = APIKeyQuery(
            name="api_key",
            auto_error=False,
            description="API key passed as query parameter (not recommended for production)",
        )

    def _verify_api_key(self, provided_key: str | None) -> bool:
        """
        Verify API key using constant-time comparison.

        Uses secrets.compare_digest to prevent timing attacks.

        Args:
            provided_key: The API key provided in the request

        Returns:
            True if key is valid, False otherwise
        """
        if not provided_key or not self.api_key:
            return False

        # Constant-time comparison to prevent timing attacks
        return secrets.compare_digest(provided_key, self.api_key)

    async def __call__(
        self,
        api_key_header: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
        api_key_query: str | None = Security(APIKeyQuery(name="api_key", auto_error=False)),
    ) -> str:
        """
        Validate API key from header or query parameter.

        Priority: Header > Query Parameter

        Args:
            api_key_header: API key from X-API-Key header
            api_key_query: API key from query parameter

        Returns:
            The validated API key

        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication if no API key is configured (development mode)
        if not self.api_key:
            logger.warning(
                "API key not configured - running in development mode without authentication"
            )
            return "development"

        # Try header first, then query parameter
        provided_key = api_key_header or api_key_query

        if not provided_key:
            logger.warning("Authentication failed: No API key provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "missing_api_key",
                        "message": "API key is required. Provide it via X-API-Key header.",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if not self._verify_api_key(provided_key):
            logger.warning("Authentication failed: Invalid API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "invalid_api_key",
                        "message": "Invalid API key provided.",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        logger.debug("Authentication successful")
        return provided_key


# Global auth instance
_api_key_auth: APIKeyAuth | None = None


def get_api_key_auth() -> APIKeyAuth:
    """Get or create the global API key auth instance."""
    global _api_key_auth
    if _api_key_auth is None:
        settings = get_settings()
        _api_key_auth = APIKeyAuth(settings)
    return _api_key_auth


async def verify_api_key(
    api_key_header: Annotated[
        str | None,
        Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
    ] = None,
    api_key_query: Annotated[
        str | None,
        Security(APIKeyQuery(name="api_key", auto_error=False)),
    ] = None,
) -> str:
    """
    FastAPI dependency for API key verification.

    Usage:
        @app.get("/protected")
        async def protected_route(api_key: str = Depends(verify_api_key)):
            ...

    Args:
        api_key_header: API key from header
        api_key_query: API key from query

    Returns:
        Validated API key

    Raises:
        HTTPException: If authentication fails
    """
    auth = get_api_key_auth()
    return await auth(api_key_header, api_key_query)


# Type alias for dependency injection
APIKeyDep = Annotated[str, Depends(verify_api_key)]
