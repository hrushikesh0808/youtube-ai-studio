"""
Story Prompt Templates

Contains prompts used by the Story Generator Agent.
"""


def build_story_prompt(topic: str) -> str:
    """
    Build a story generation prompt.
    """

    return f"""
You are a professional YouTube storyteller.

Write a highly engaging YouTube story about:

Topic:
{topic}

Requirements:

- Hook the audience in the first paragraph.
- Keep the story emotional.
- Use simple English.
- Make the story suitable for narration.
- Include suspense where appropriate.
- End with a satisfying conclusion.

Return only the story.
"""