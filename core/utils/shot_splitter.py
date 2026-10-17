
"""
Shot Splitter

Splits an approved narration script into timed shots.

This is deliberately deterministic (no LLM): a language model asked to chop
a story into shots tends to paraphrase, skip or reorder text. Here the
narration of the shots, joined in order, is always exactly the story.
"""

import re

from core.exceptions.exceptions import WorkflowError

# A sentence ends at . ! or ? (plus any closing quotes/brackets) when the
# next sentence starts with a capital letter, digit, or an opening quote.
_SENTENCE_END = re.compile(r'[.!?]["”’\')\]]*(?=\s+["“‘\'(\[]?[A-Z0-9])')

# Words that end in a full stop without ending a sentence.
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "jr.", "sr."}
_INITIALS = re.compile(r"^(?:[A-Za-z]\.)+$")


def split_sentences(paragraph: str) -> list[str]:
    """
    Split one paragraph into sentences without altering any text.
    """
    sentences = []
    start = 0

    for match in _SENTENCE_END.finditer(paragraph):
        end = match.end()
        piece = paragraph[start:end].strip()
        last_word = piece.split()[-1]

        if last_word.lower() in _ABBREVIATIONS or _INITIALS.match(last_word):
            continue

        sentences.append(piece)
        start = end

    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)

    return sentences


def _allocate_shots(
    paragraph_words: list[int],
    sentence_counts: list[int],
    total_shots: int,
) -> list[int]:
    """
    Decide how many shots each paragraph gets, in proportion to its length.
    Every paragraph gets at least 1 shot and at most one shot per sentence.
    """
    total_words = sum(paragraph_words)
    ideal = [total_shots * w / total_words for w in paragraph_words]
    alloc = [
        min(max(int(share), 1), count)
        for share, count in zip(ideal, sentence_counts)
    ]

    while sum(alloc) < total_shots:
        candidates = [i for i in range(len(alloc)) if alloc[i] < sentence_counts[i]]
        if not candidates:
            break
        i = max(candidates, key=lambda i: ideal[i] - alloc[i])
        alloc[i] += 1

    while sum(alloc) > total_shots:
        candidates = [i for i in range(len(alloc)) if alloc[i] > 1]
        if not candidates:
            break
        i = min(candidates, key=lambda i: ideal[i] - alloc[i])
        alloc[i] -= 1

    return alloc


def _group_sentences(sentence_words: list[int], groups: int) -> list[tuple[int, int]]:
    """
    Split sentences into `groups` consecutive runs with balanced word counts.
    Returns (start, end) index pairs, end exclusive.
    """
    n = len(sentence_words)

    if groups >= n:
        return [(i, i + 1) for i in range(n)]

    prefix = [0]
    for w in sentence_words:
        prefix.append(prefix[-1] + w)

    target = prefix[-1] / groups
    inf = float("inf")

    cost = [[inf] * (n + 1) for _ in range(groups + 1)]
    back = [[0] * (n + 1) for _ in range(groups + 1)]
    cost[0][0] = 0.0

    for g in range(1, groups + 1):
        for i in range(g, n + 1):
            for j in range(g - 1, i):
                value = cost[g - 1][j] + (prefix[i] - prefix[j] - target) ** 2
                if value < cost[g][i]:
                    cost[g][i] = value
                    back[g][i] = j

    bounds = []
    i = n
    for g in range(groups, 0, -1):
        j = back[g][i]
        bounds.append((j, i))
        i = j

    return list(reversed(bounds))


def build_shot_plan(
    source: str,
    text: str,
    words_per_minute: int,
    target_shot_seconds: int,
    min_shots: int,
    max_shots: int,
) -> dict:
    """
    Split narration text into timed shots and return the shot plan.

    Shots never span a paragraph break, and the narration of all shots,
    joined in order, is exactly the original text.

    Raises:
        WorkflowError: if the text is empty or the split loses any text.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]

    if not paragraphs:
        raise WorkflowError("Cannot split an empty story into shots.")

    paragraph_sentences = [split_sentences(p) for p in paragraphs]
    sentence_counts = [len(s) for s in paragraph_sentences]
    paragraph_words = [
        sum(len(sentence.split()) for sentence in sentences)
        for sentences in paragraph_sentences
    ]

    total_words = sum(paragraph_words)
    total_seconds = total_words / words_per_minute * 60

    desired = round(total_seconds / target_shot_seconds)
    desired = max(min_shots, min(max_shots, desired))
    desired = max(desired, len(paragraphs))
    desired = min(desired, sum(sentence_counts))

    allocation = _allocate_shots(paragraph_words, sentence_counts, desired)

    shots = []
    words_so_far = 0

    for p_index, sentences in enumerate(paragraph_sentences):
        sentence_words = [len(s.split()) for s in sentences]

        for start, end in _group_sentences(sentence_words, allocation[p_index]):
            narration = " ".join(sentences[start:end])
            words = sum(sentence_words[start:end])

            shots.append(
                {
                    "id": len(shots) + 1,
                    "paragraph": p_index + 1,
                    "narration": narration,
                    "words": words,
                    "start_seconds": round(words_so_far / words_per_minute * 60, 1),
                    "duration_seconds": round(words / words_per_minute * 60, 1),
                }
            )
            words_so_far += words

    rebuilt = " ".join(shot["narration"] for shot in shots).split()

    if rebuilt != text.split():
        raise WorkflowError("Shot split changed or lost narration text.")

    return {
        "source": source,
        "words_per_minute": words_per_minute,
        "target_shot_seconds": target_shot_seconds,
        "total_words": total_words,
        "total_seconds": round(total_seconds, 1),
        "shot_count": len(shots),
        "shots": shots,
    }

