"""
Story Generator Agent
"""

from core.config.settings import settings
from core.utils.text_cleaner import clean_narration_text
from prompts.prompt_loader import PromptLoader
from services.llm.ollama_provider import OllamaProvider


class StoryGenerator:
    """
    Generates stories using the configured AI provider.
    """

    def __init__(self):
        self.provider = OllamaProvider()
        self.prompt_loader = PromptLoader()

    def generate(self, topic: str) -> str:
        """
        Generate a story from a topic and return it as plain narration text.
        """

        prompt = self.prompt_loader.load(
            "story.txt",
            topic=topic,
            min_words=settings.target_min_words,
            max_words=settings.target_max_words,
        )

        return clean_narration_text(self.provider.generate(prompt))