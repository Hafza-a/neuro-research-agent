"""
Journal Reviewer agent — evaluates the draft as a peer reviewer for a top neuroscience journal.
Returns structured feedback and a numerical score (1–10). No tool use.
"""
import re
import anthropic
from prompts.reviewer_system import REVIEWER_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


def _parse_score(reviewer_text: str) -> int:
    """
    Extract the numerical score from the reviewer's VERDICT line.
    Falls back to keyword matching if the structured line is missing.
    """
    # Primary: look for "VERDICT: X/10"
    match = re.search(r"VERDICT:\s*(\d+)/10", reviewer_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(1, min(10, score))  # clamp to [1, 10]

    # Fallback: parse verdict keywords
    lower = reviewer_text.lower()
    if "accept" in lower and "revision" not in lower and "reject" not in lower:
        return 9
    if "minor revision" in lower:
        return 7
    if "major revision" in lower:
        return 5
    if "reject" in lower:
        return 3
    return 6  # Unknown — assume needs work


def reviewer_critique(
    client: anthropic.Anthropic,
    draft: str,
) -> tuple:
    """
    Journal reviewer evaluates the draft.

    Returns:
        feedback_text (str): full structured review
        score (int): numerical score 1–10
    """
    prompt = f"""Please review the following literature review manuscript submitted to your journal.
Apply your full peer-review standards.

---
## MANUSCRIPT:
{draft}
---

Provide your complete peer review following your instructions. The VERDICT line is mandatory."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=REVIEWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    feedback = "".join(b.text for b in response.content if b.type == "text")
    score = _parse_score(feedback)
    return feedback, score
