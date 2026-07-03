"""
Prompt Loader

Loads prompt templates and injects variables.
"""

from pathlib import Path


class PromptLoader:
    """
    Loads prompt templates from the templates directory.
    """

    TEMPLATE_DIR = Path(__file__).parent / "templates"

    def load(self, template_name: str, **kwargs) -> str:
        """
        Load a prompt template and replace variables.
        """

        template_path = self.TEMPLATE_DIR / template_name

        with open(template_path, "r", encoding="utf-8") as file:
            template = file.read()

        return template.format(**kwargs)