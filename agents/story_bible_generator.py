
"""
Story Bible Generator Agent

Turns an approved story into a validated story bible (characters,
locations, props) using the configured AI provider.
"""

from core.config.settings import settings
from core.exceptions.exceptions import AgentError
from core.exceptions.exceptions import ValidationError
from core.logging.logger import logger
from core.utils.story_bible import bible_warnings
from core.utils.story_bible import extract_json
from core.utils.story_bible import validate_bible
from prompts.prompt_loader import PromptLoader
from services.llm.ollama_provider import OllamaProvider


class StoryBibleGenerator:
    """
    Generates and validates the story bible.

    Invalid replies are retried a bounded number of times, and the
    problems found are sent back to the model so it can correct them.
    """

    def __init__(self, provider=None):
        self.provider = provider or OllamaProvider()
        self.prompt_loader = PromptLoader()

    def generate(self, story: str) -> dict:
        """
        Return a validated story bible for the story.

        Raises:
            AgentError: if no valid bible is produced within the allowed attempts.
        """
        attempts = settings.bible_max_attempts
        base_prompt = self.prompt_loader.load("bible.txt", story=story)
        prompt = base_prompt
        last_problem = ""

        for attempt in range(1, attempts + 1):
            logger.info("Story bible: attempt %d of %d", attempt, attempts)

            raw = self.provider.generate(prompt, json_mode=True)

            try:
                bible = validate_bible(extract_json(raw))
            except ValidationError as exc:
                last_problem = str(exc)
                logger.warning("Story bible attempt %d rejected: %s", attempt, last_problem)

                prompt = (
                    base_prompt
                    + "\n\nYour previous answer was rejected for these reasons:\n"
                    + last_problem
                    + "\n\nReturn the complete corrected JSON object only."
                )
                continue

            for warning in bible_warnings(bible, story):
                logger.warning(warning)

            return {"art_style": settings.art_style, **bible}

        raise AgentError(
            f"Story bible failed validation after {attempts} attempts. "
            f"Last problem: {last_problem}"
        )
