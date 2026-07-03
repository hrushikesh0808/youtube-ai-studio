"""
Story Generator Agent
"""

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
        Generate a story from a topic.
        """

        prompt = self.prompt_loader.load(
            "story.txt",
            topic=topic,
        )

        return self.provider.generate(prompt)