"""
Text Cleaner

Turns raw LLM output into plain narration text: only the words a voice
actor would speak. Small models often add chatty preambles, markdown,
and stage directions such as "(Pause - 3 seconds)" even when told not to,
and a text-to-speech voice would read all of that aloud.
"""

import re

# Chatty first line, e.g. "Okay, here's the script:"
_PREAMBLE_STARTS = (
    "okay",
    "ok,",
    "sure",
    "certainly",
    "alright",
    "here's",
    "here is",
)

# A whole line that is only a parenthetical, e.g. "**(Intro Music fades in)**"
_DIRECTION_LINE = re.compile(r"^\s*\**\s*\(.*\)\s*\**\s*$")

# Inline parenthetical that is clearly an audio/pacing direction.
_INLINE_DIRECTION = re.compile(
    r"\s*\((?=[^()]*\b(?:music|pause|sound|sfx|fade|fades|swell|swells|"
    r"silence|beat|cue|seconds?)\b)[^()]*\)",
    re.IGNORECASE,
)

# A full stop / ! / ? glued to the next sentence: "stopped.The sun"
_GLUED_SENTENCE = re.compile(r"(?<=[a-z][.!?])(?=[A-Z])")


def clean_narration_text(text: str) -> str:
    """
    Return plain narration text with directions and markdown removed.
    """

    lines = text.strip().splitlines()

    # 1. Remove a leading chatty preamble line.
    if lines:
        first = lines[0].strip().lower()
        if first.startswith(_PREAMBLE_STARTS) and first.endswith(":"):
            lines = lines[1:]

    cleaned = []

    for line in lines:
        # 2. Drop lines that are purely a stage direction.
        if _DIRECTION_LINE.match(line):
            continue

        # 3. Drop bare separators such as --- or ***.
        stripped = line.strip()
        if stripped and set(stripped) <= set("-*_"):
            continue

        # 4. Remove inline audio/pacing directions.
        line = _INLINE_DIRECTION.sub("", line)

        # 5. Remove markdown emphasis and heading markers.
        line = line.replace("*", "")
        line = re.sub(r"^\s*#+\s*", "", line)

        # 6. Fix a sentence glued to the next one.
        line = _GLUED_SENTENCE.sub(" ", line)

        cleaned.append(line.rstrip())

    result = "\n".join(cleaned)

    # 7. Collapse runs of blank lines.
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()

