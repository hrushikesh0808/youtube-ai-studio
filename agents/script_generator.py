"""
Script Generator Agent

Generates a YouTube narration script from a story.
"""

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
        """

        prompt = self.loader.load(
            "script.txt",
            story=story,
        )

        return self.provider.generate(prompt)