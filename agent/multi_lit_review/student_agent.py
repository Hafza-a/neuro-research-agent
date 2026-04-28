"""
PhD Student agent — writes the initial draft, revisions, and final polished version.
Works exclusively from the provided paper pool (no tool use) to prevent hallucinations.
"""
import anthropic
from prompts.student_system import STUDENT_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


def _format_papers(papers: list) -> str:
    """Format the paper pool as numbered context for the student."""
    lines = []
    for i, p in enumerate(papers, 1):
        title   = p.get("title", "Unknown title")
        authors = p.get("authors", "Unknown")
        year    = p.get("year", "n.d.")
        source  = p.get("source", "")
        abstract = (p.get("abstract") or "")[:450]
        doi_url = p.get("doi") or p.get("url", "")

        lines.append(f"[{i}] **{title}**")
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
    """
    Call Claude and transparently continue if the response hits max_tokens.
    Returns the full concatenated text.
    """
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

        # Append the partial output and ask Claude to continue seamlessly
        msgs.append({"role": "assistant", "content": chunk})
        msgs.append({
            "role": "user",
            "content": (
                "Continue exactly where you left off. "
                "Do not repeat any content already written. "
                "Resume immediately from the last word."
            ),
        })

    return full_text


# ── Public API ─────────────────────────────────────────────────────────────────

def student_write_initial(
    client: anthropic.Anthropic,
    research_question: str,
    paper_type: str,
    papers: list,
    n_raw: int,
    n_deduped: int,
) -> str:
    """PhD student writes the first complete draft from the paper pool."""
    papers_text = _format_papers(papers)

    prompt = f"""Write a **{paper_type}** on the following research question:

**Research Question:** {research_question}

**Search Statistics (use these exact numbers in the Methods section):**
- Databases searched: PubMed, Semantic Scholar, arXiv, bioRxiv
- Total records retrieved: {n_raw}
- Records after deduplication: {n_deduped}
- Papers included in this review: {len(papers)}

**Your Paper Pool — you may ONLY cite papers from this list:**
{papers_text}

Now write the complete {paper_type}. Follow the correct structure for this paper type. Cite papers as (First Author et al., Year). Include a full numbered reference list at the end."""

    return _call_with_continuation(
        client,
        STUDENT_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
        max_tokens=6000,
        max_passes=3,
    )


def student_revise(
    client: anthropic.Anthropic,
    draft: str,
    sup_feedback: str,
    peer_feedback: str,
    rev_feedback: str,
    research_question: str,
    all_papers: list,
) -> str:
    """Student revises the draft incorporating feedback from all three reviewers."""
    papers_text = _format_papers(all_papers)

    prompt = f"""You are revising your literature review on:
**"{research_question}"**

Three reviewers have provided feedback below. Incorporate ALL of their points.

---
### SUPERVISOR FEEDBACK:
{sup_feedback}

---
### PEER FEEDBACK (includes papers found via additional searches):
{peer_feedback}

---
### JOURNAL REVIEWER FEEDBACK:
{rev_feedback}

---
### YOUR CURRENT DRAFT:
{draft}

---
### UPDATED PAPER POOL (includes any new papers found by your peer — cite from this list only):
{papers_text}

Write the complete revised literature review now. Address every piece of feedback. Do not introduce citations that are not in the paper pool above."""

    return _call_with_continuation(
        client,
        STUDENT_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
        max_tokens=6000,
        max_passes=3,
    )


def student_final_polish(
    client: anthropic.Anthropic,
    draft: str,
    sup_feedback: str,
    peer_feedback: str,
    rev_feedback: str,
    research_question: str,
    all_papers: list,
) -> str:
    """Student writes the final, publication-ready version."""
    papers_text = _format_papers(all_papers)

    prompt = f"""You are writing the **final, publication-ready** version of your literature review on:
**"{research_question}"**

This is your last revision. Make it truly excellent — address every point from every reviewer, and ensure the writing is polished, the synthesis is critical, and every claim is properly cited.

---
### SUPERVISOR FEEDBACK:
{sup_feedback}

---
### PEER FEEDBACK:
{peer_feedback}

---
### JOURNAL REVIEWER FEEDBACK:
{rev_feedback}

---
### YOUR CURRENT DRAFT:
{draft}

---
### PAPER POOL (cite ONLY from this list):
{papers_text}

Write the **complete, final, polished** version now. Ensure:
- Every factual claim is cited from the paper pool
- The synthesis is thematic and critical, not paper-by-paper
- The research gaps section is specific and grounded in the evidence
- The reference list is complete and accurate
- The writing is clear, precise, and publication-ready throughout"""

    return _call_with_continuation(
        client,
        STUDENT_SYSTEM_PROMPT,
        [{"role": "user", "content": prompt}],
        max_tokens=6000,
        max_passes=3,
    )
