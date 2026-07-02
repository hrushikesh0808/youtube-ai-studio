"""
Custom Application Exceptions

Defines all project-specific exceptions.
"""


class YouTubeAIStudioError(Exception):
    """
    Base exception for the entire application.
    """

    def __init__(self, message: str):
        super().__init__(message)


class ConfigurationError(YouTubeAIStudioError):
    """Raised when configuration is invalid."""


class ValidationError(YouTubeAIStudioError):
    """Raised when validation fails."""


class ProviderError(YouTubeAIStudioError):
    """Raised when an AI provider fails."""


class AgentError(YouTubeAIStudioError):
    """Raised when an AI agent fails."""


class WorkflowError(YouTubeAIStudioError):
    """Raised when workflow execution fails."""


class MemoryError(YouTubeAIStudioError):
    """Raised when memory operations fail."""


class VideoGenerationError(YouTubeAIStudioError):
    """Raised when video generation fails."""


class VoiceGenerationError(YouTubeAIStudioError):
    """Raised when voice generation fails."""


class ImageGenerationError(YouTubeAIStudioError):
    """Raised when image generation fails."""


class QAError(YouTubeAIStudioError):
    """Raised when YouTube QA validation fails."""