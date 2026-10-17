"""
Ollama Provider

Provides communication with the local Ollama server.
"""

from typing import Any

import requests

from core.base.provider import BaseProvider
from core.config.settings import settings
from core.exceptions.exceptions import ProviderError
from services.network.http_client import HttpClient


class OllamaProvider(BaseProvider):
    """
    Ollama LLM Provider.
    """

    def __init__(self):
        self.client = HttpClient()
        self.base_url = settings.ollama_host

    @property
    def provider_name(self) -> str:
        """
        Return provider name.
        """
        return "Ollama"

    def initialize(self) -> None:
        """
        Initialize the provider.
        """
        pass

    def is_available(self) -> bool:
        """
        Check whether Ollama is available.
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")

            return response.status_code == 200

        except Exception:
            return False

    def list_models(self) -> list[str]:
        """
        Return all installed Ollama models.
        """
        response = self.client.get(f"{self.base_url}/api/tags")

        response.raise_for_status()

        data = response.json()

        return [
            model["name"]
            for model in data.get("models", [])
        ]

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        """
        Generate text using Ollama.

        json_mode=True constrains the reply to valid JSON of any shape.
        json_schema=<JSON Schema dict> constrains it to that structure
        (needs Ollama 0.5 or newer) and takes priority over json_mode.
        The prompt must still ask for JSON in both cases.

        Raises:
            ProviderError: if Ollama cannot be reached or returns an error.
        """
        payload = {
            "model": model or settings.default_model,
            "prompt": prompt,
            "stream": False,
        }

        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"

        if settings.ollama_temperature is not None:
            payload["options"] = {"temperature": settings.ollama_temperature}

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc

        if not response.ok:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: "
                f"{response.text.strip()[:300]}"
            )

        data = response.json()

        return data["response"]