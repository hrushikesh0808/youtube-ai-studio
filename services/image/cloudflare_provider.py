"""
Cloudflare Image Provider

Generates images with FLUX.1 schnell on Cloudflare Workers AI (free tier:
10,000 neurons per day). Credentials come only from the .env file.

What the service accepts, as found in testing: a prompt (up to 2048
characters) and a step count (1 to 8). It rejects seed, width and height,
and returns a fixed-size square picture.
"""

import base64
import json
import time

import requests

from core.base.provider import BaseProvider
from core.config.settings import settings
from core.exceptions.exceptions import ConfigurationError
from core.exceptions.exceptions import ImageGenerationError
from core.logging.logger import logger
from services.network.http_client import HttpClient

MAX_PROMPT_LENGTH = 2048

REQUEST_TIMEOUT_SECONDS = 120

# Only temporary problems are retried; a wrong token or a bad request is not.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5

STATUS_HINTS = {
    400: "The service rejected the request.",
    401: "The API token is wrong or expired.",
    403: "The token lacks Workers AI permission.",
    404: "The account ID or model name is wrong.",
    429: "The free daily allowance may be used up; it resets at 00:00 UTC.",
}


class CloudflareImageProvider(BaseProvider):
    """Image generation on Cloudflare Workers AI."""

    def __init__(self, http_client: HttpClient | None = None, sleep=time.sleep):
        self.account_id = settings.cloudflare_account_id
        self.token = settings.cloudflare_api_token
        self.model = settings.cloudflare_image_model
        self.api_base = settings.cloudflare_api_base.rstrip("/")
        self.http = http_client or HttpClient()
        self.sleep = sleep

    @property
    def provider_name(self) -> str:
        return "cloudflare"

    @property
    def model_name(self) -> str:
        return self.model

    def initialize(self) -> None:
        """
        Check that the credentials are configured.

        Raises:
            ConfigurationError: if the account ID or token is missing from .env.
        """
        missing = [
            name
            for name, value in (
                ("CLOUDFLARE_ACCOUNT_ID", self.account_id),
                ("CLOUDFLARE_API_TOKEN", self.token),
            )
            if not value
        ]

        if missing:
            raise ConfigurationError("Missing in .env: " + ", ".join(missing))

    def is_available(self) -> bool:
        """True when credentials are configured (no network call is made)."""
        return bool(self.account_id and self.token)

    def generate(self, prompt: str, steps: int | None = None) -> bytes:
        """
        Generate one picture and return its bytes.

        Raises:
            ImageGenerationError: if the prompt is unusable or the request fails.
        """
        if not prompt.strip() or len(prompt) > MAX_PROMPT_LENGTH:
            raise ImageGenerationError(f"The prompt must be 1 to {MAX_PROMPT_LENGTH} characters.")

        url = f"{self.api_base}/accounts/{self.account_id}/ai/run/{self.model}"
        payload = {"prompt": prompt}

        if steps is not None:
            payload["steps"] = steps

        headers = {"Authorization": f"Bearer {self.token}"}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.http.post(
                    url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    headers=headers,
                )
            except requests.RequestException as exc:
                if attempt < MAX_ATTEMPTS:
                    self._wait("Network problem", attempt)
                    continue
                raise ImageGenerationError(f"Network problem: {type(exc).__name__}") from exc

            if response.status_code < 400:
                return self._extract_image(response)

            if response.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                self._wait(f"HTTP {response.status_code}", attempt)
                continue

            hint = STATUS_HINTS.get(response.status_code, "")
            detail = response.text[:300]
            raise ImageGenerationError(f"HTTP {response.status_code}. {hint} {detail}".strip())

        raise ImageGenerationError("The image request failed.")  # not reachable

    def _wait(self, reason: str, attempt: int) -> None:
        wait = RETRY_WAIT_SECONDS * attempt
        logger.warning("%s, retrying in %d s (attempt %d/%d)", reason, wait, attempt, MAX_ATTEMPTS)
        self.sleep(wait)

    @staticmethod
    def _extract_image(response) -> bytes:
        """Return image bytes from a raw image reply or a JSON reply with base64."""
        content_type = response.headers.get("Content-Type", "").lower()

        if content_type.startswith("image/"):
            return response.content

        try:
            data = json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImageGenerationError("The reply was neither an image nor JSON.") from exc

        if isinstance(data, dict) and data.get("success") is False:
            errors = data.get("errors") or []
            messages = "; ".join(str(item.get("message", item)) for item in errors if item)
            raise ImageGenerationError(f"Cloudflare reported an error: {messages or 'unknown'}")

        encoded = None

        if isinstance(data, dict):
            result = data.get("result")

            if isinstance(result, dict):
                encoded = result.get("image")

            if encoded is None:
                encoded = data.get("image")

        if not encoded or not isinstance(encoded, str):
            raise ImageGenerationError("The reply contained no image field.")

        try:
            return base64.b64decode(encoded)
        except ValueError as exc:
            raise ImageGenerationError("The image data could not be decoded.") from exc