"""
Script Generator Agent

Generates a YouTube narration script from a story.
"""

import requests

from core.config.settings import settings
from core.exceptions.exceptions import AgentError
from core.exceptions.exceptions import ProviderError
from core.utils.text_cleaner import clean_narration_text
from prompts.prompt_loader import PromptLoader
from services.llm.ollama_provider import OllamaProvider


class ScriptGenerator:
    """
    Generates narration scripts from stories.
    """

    def __init__(self):
        self.loader = PromptLoader()
        self.provider = OllamaProvider()

    def generate(self, story: str) -> str:
        """
        Generate narration script.

        Raises:
            ProviderError: if the LLM provider cannot be reached or fails.
            AgentError: if the model returns no usable script.
        """

        prompt = self.loader.load(
            "script.txt",
            story=story,
            min_words=settings.target_min_words,
            max_words=settings.target_max_words,
        )

        try:
            raw = self.provider.generate(prompt)

        except requests.exceptions.Timeout as exc:
            raise ProviderError(
                "Ollama timed out while generating the script."
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise ProviderError(
                f"Ollama request failed while generating the script: {exc}"
            ) from exc

        script = clean_narration_text(raw)

        if not script:
            raise AgentError("Script generator returned empty output.")

        return script