import anthropic
from prompts.hypothesis_system import HYPOTHESIS_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


def generate_hypotheses(
    client: anthropic.Anthropic,
    review_text: str,
    research_question: str,
) -> str:
    """
    Given a completed literature review, generate 3-5 concrete, fundable
    research hypotheses grounded in the gaps identified in the review.
    """
    # Extract just the gaps section if present, otherwise use the full review
    gaps_section = review_text
    for marker in ["## 5. Research Gaps", "## Research Gaps", "## 6. Research Gaps",
                   "## Future Directions", "## Gaps"]:
        if marker in review_text:
            gaps_section = review_text[review_text.index(marker):]
            break

    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=HYPOTHESIS_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""Based on the following systematic literature review on "{research_question}", generate 3-5 concrete, fundable research hypotheses grounded in the specific gaps identified.

Each hypothesis must be directly traceable to a gap named in the review.
Be specific about model systems, techniques, sample sizes, and timelines.

--- REVIEW CONTENT (gaps section / full review) ---
{gaps_section[:12000]}
--- END ---

Generate the Hypothesis Report now.""",
        }],
    )

    hypotheses_text = response.content[0].text

    # Handle continuation if needed
    if response.stop_reason == "max_tokens":
        cont = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=HYPOTHESIS_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Generate hypotheses for: {research_question}"},
                {"role": "assistant", "content": hypotheses_text},
                {"role": "user", "content": "Continue from exactly where you left off."},
            ],
        )
        hypotheses_text += cont.content[0].text

    return hypotheses_text
