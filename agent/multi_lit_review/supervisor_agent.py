"""
Supervisor agent — reviews the student's draft and returns structured, actionable feedback.
No tool use. Works from the draft text alone.
"""
import anthropic
from prompts.supervisor_system import SUPERVISOR_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


def supervisor_review(
    client: anthropic.Anthropic,
    draft: str,
    research_question: str,
) -> str:
    """
    Supervisor reads the student's draft and returns detailed feedback.
    Returns the feedback as a markdown string.
    """
    prompt = f"""Your PhD student has submitted a draft literature review on:
**"{research_question}"**

Please review it carefully and provide detailed, actionable feedback following your instructions.

---
## DRAFT TO REVIEW:
{draft}
---

Provide your structured supervisor feedback now."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=SUPERVISOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return "".join(b.text for b in response.content if b.type == "text")
