"""
Image Provider Factory

Chooses the image provider named by IMAGE_PROVIDER in .env, so a different
or stronger provider can be added later without touching the pipeline.
"""

from core.base.provider import BaseProvider
from core.config.settings import settings
from core.exceptions.exceptions import ConfigurationError
from services.image.cloudflare_provider import CloudflareImageProvider

SUPPORTED_PROVIDERS = ("cloudflare",)


def create_image_provider() -> BaseProvider:
    """
    Return the configured image provider (not yet initialized).

    Raises:
        ConfigurationError: if IMAGE_PROVIDER names an unknown provider.
    """
    if settings.image_provider == "cloudflare":
        return CloudflareImageProvider()

    raise ConfigurationError(
        f"IMAGE_PROVIDER '{settings.image_provider}' is not supported. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def image_uses_references() -> bool:
    """True when the configured image model takes reference pictures (FLUX.2 on Cloudflare)."""
    return settings.image_provider == "cloudflare" and "flux-2" in settings.cloudflare_image_model
