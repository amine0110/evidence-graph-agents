"""The decision every C3 system makes, worded once so S1 to S4 get the same task.

It mirrors the question the human annotators answered: does this clip need a
security operator's attention? The answer format gives a score (the
probability of "Yes" on the first token) and time segments for grounding.
"""

import re

TASK_VERSION = "task-v1"
QUESTION = (
    "You are assisting a security operator who monitors video. Decide whether this clip needs the "
    "operator's attention because it shows violence or danger: fighting, shooting, a riot, abuse, a car "
    "accident or an explosion. Films and news footage are included; judge what is shown, not whether it "
    "is staged.\n"
    "Answer on the first line with Yes or No only. On the second line, give the time segments in seconds "
    "where it happens, like 12-20, 31-40, or write none.")


def parse_answer(text):
    """(decision or None, [(start, end), ...]) from a model's answer."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    first = lines[0].lower() if lines else ""
    decision = "yes" if first.startswith("yes") else "no" if first.startswith("no") else None
    rest = " ".join(lines[1:])
    segments = [(float(a), float(b)) for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", rest)]
    return decision, [(a, b) for a, b in segments if b >= a]
