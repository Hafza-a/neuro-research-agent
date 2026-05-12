"""
Writer agent — produces the literature review from the enriched paper pool
and the Scout's field briefing. Revises when given adversarial critique.
No tool use: works exclusively from provided context to prevent hallucinations.
"""
import anthropic
from prompts.writer_system import WRITER_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


def _format_papers(papers: list) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        title    = p.get("title", "Unknown title")
        authors  = p.get("authors", "Unknown")
        year     = p.get("year", "n.d.")
        source   = p.get("source", "")
        abstract = (p.get("abstract") or "")[:450]
        doi_url  = p.get("doi") or p.get("url", "")

        lines.append(f"[{i}] {title}")
        lines.append(f"    {authors} ({year}) — {source}")
        if abstract:
            lines.append(f"    {abstract}")
        if doi_url:
            lines.append(f"    {doi_url}")
        lines.append("")
    return "\n".join(lines)


def _call_with_continuation(
    client: anthropic.Anthropic,
    system: str,
    messages: list,
    max_tokens: int = 6000,
    max_passes: int = 3,
) -> str:
    full_text = ""
    msgs = list(messages)
    for _ in range(max_passes):
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=msgs,
        )
        chunk = "".join(b.text for b in response.content if b.type == "text")
        full_text += chunk
        if response.stop_reason != "max_tokens":
            break
        msgs.append({"role": "assistant", "content": chunk})
        msgs.append({
            "role": "user",
            "content": (
                "Continue exactly where you left off. "
                "Do not repeat anything already written. "
                "Resume from the last word immediately."
            ),
        })
    return full_text


# ── Public API ─────────────────────────────────────────────────────────────────

def writer_draft(
    client: anthropic.Anthropic,
    research_question: str,
    paper_type: str,
    papers: list,
    scout_briefing: str,
    n_raw: int,
    n_deduped: int,
) -> str:
    papers_text = _format_papers(papers)
    prompt = f"""Write a **{paper_type}** on:
**"{research_question}"**

**Search statistics for the Methods section (use these exact numbers):**
- Databases: PubMed, Semantic Scholar, arXiv, bioRxiv
- Records retrieved: {n_raw}
- After deduplication: {n_deduped}
- Papers included: {len(papers)}

**Scout's Field Briefing (use this to orient your synthesis):**
{scout_briefing}

**Paper Pool — cite ONLY from this list:**
{papers_text}

Write the complete {paper_type} now. Use (First Author et al., Year) citations inline. Include a numbered reference list at the end."""

    return _call_with_continuation(
        client, WRITER_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
        max_tokens=6000, max_passes=3,
    )


def writer_revise(
    client: anthropic.Anthropic,
    research_question: str,
    papers: list,
    scout_briefing: str,
    draft: str,
    critique: str,
) -> str:
    papers_text = _format_papers(papers)
    prompt = f"""You are revising your literature review on:
**"{research_question}"**

The adversarial reviewer has critiqued your draft. Address every numbered point.

---
**ADVERSARIAL CRITIQUE:**
{critique}

---
**YOUR CURRENT DRAFT:**
{draft}

---
**UPDATED PAPER POOL (includes any papers found by the reviewer — cite from this list only):**
{papers_text}

**Scout's Field Briefing (for reference):**
{scout_briefing}

Write the complete revised literature review now. Do not introduce citations that are not in the paper pool above."""

    return _call_with_continuation(
        client, WRITER_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
        max_tokens=6000, max_passes=3,
    )


def writer_final_polish(
    client: anthropic.Anthropic,
    research_question: str,
    papers: list,
    scout_briefing: str,
    draft: str,
    critique: str,
) -> str:
    papers_text = _format_papers(papers)
    prompt = f"""Write the **final, publication-ready** version of your literature review on:
**"{research_question}"**

This is your last pass. Make it genuinely excellent — precise, well-synthesised, every claim cited, every gap specific.

---
**ADVERSARIAL CRITIQUE TO ADDRESS:**
{critique}

---
**YOUR CURRENT DRAFT:**
{draft}

---
**PAPER POOL (cite ONLY from this list):**
{papers_text}

Write the complete, polished final version now. Ensure:
- Every factual claim is cited from the pool
- The synthesis is thematic and critical, not descriptive
- The gaps section is specific and grounded in the evidence
- The reference list is complete and accurate"""

    return _call_with_continuation(
        client, WRITER_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
        max_tokens=6000, max_passes=3,
    )
