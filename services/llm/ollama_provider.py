"""
Ollama Provider

Provides communication with the local Ollama server.
"""

from typing import Any

from core.base.provider import BaseProvider
from core.config.settings import settings
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

    def generate(self, prompt: str, model: str | None = None) -> str:
        """
        Generate text using Ollama.
        """
        payload = {
            "model": model or settings.default_model,
            "prompt": prompt,
            "stream": False,
        }

        response = self.client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        
        response.raise_for_status()

        data = response.json()

        return data["response"]