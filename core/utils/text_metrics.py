"""
Text Metrics

Helpers for measuring narration length.
"""


def count_words(text: str) -> int:
    """
    Count whitespace-separated words.
    """
    return len(text.split())


def estimate_narration_seconds(text: str, words_per_minute: int) -> int:
    """
    Estimate how long the text takes to narrate, in whole seconds.
    """
    return round(count_words(text) / words_per_minute * 60)