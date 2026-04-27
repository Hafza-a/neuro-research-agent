import asyncio
import json
from dataclasses import dataclass
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from agent.tools.arxiv_search import search_arxiv
from agent.tools.biorxiv import search_biorxiv
from agent.tools.crossref import verify_doi
from prompts.litreview_system import LITREVIEW_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"


@dataclass
class LitReviewProgress:
    phase: str
    message: str
    papers_found: int = 0


# ─── Internal helpers ──────────────────────────────────────────────────────────

async def _run_all_searches(query: str, max_per_db: int = 10) -> list[dict]:
    results = await asyncio.gather(
        search_pubmed(query, max_per_db),
        search_semantic_scholar(query, max_per_db),
        search_arxiv(query, max_per_db),
        search_biorxiv(query, 5),
        return_exceptions=True,
    )
    combined = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)
    return combined


def _deduplicate(papers: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in papers:
        key = p.get("title", "").lower().strip()[:60]
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ─── Public API ────────────────────────────────────────────────────────────────

def plan_and_search(
    client: anthropic.Anthropic,
    research_question: str,
    max_per_db: int = 10,
    progress_callback=None,
) -> tuple[dict, list, list]:
    """
    Phase 1 + 2: plan search strategy and query all databases.
    Returns (plan, all_papers_raw, deduplicated_papers).
    """
    def _progress(phase, message, n=0):
        if progress_callback:
            progress_callback(LitReviewProgress(phase, message, n))

    _progress("Planning", "Defining research scope and generating search queries…")
    plan_response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""You are planning a systematic literature review on: "{research_question}"

CRITICAL: Every search query MUST stay true to the exact scope the user asked for.
If the topic is broad (e.g. "AI in neuroscience"), keep queries varied across sub-areas.
If specific (e.g. "astrocytes in synaptic pruning"), stay tightly focused.

Generate a JSON object with:
1. "pico": object with keys population, intervention, comparison, outcome (omit irrelevant ones)
2. "search_queries": list of 4 varied search query strings covering the FULL BREADTH of the topic
3. "inclusion_criteria": list of 3-5 strings
4. "exclusion_criteria": list of 2-4 strings

Respond with ONLY valid JSON, no markdown fences."""
        }],
    )

    try:
        plan = json.loads(plan_response.content[0].text)
    except Exception:
        plan = {
            "pico": {},
            "search_queries": [research_question],
            "inclusion_criteria": ["Peer-reviewed studies", "Neuroscience-related"],
            "exclusion_criteria": ["Non-English", "Conference abstracts only"],
        }

    _progress("Searching", f"Running {len(plan['search_queries'])} queries across 4 databases…")

    all_papers = []
    for q in plan["search_queries"]:
        batch = asyncio.run(_run_all_searches(q, max_per_db))
        all_papers.extend(batch)

    papers = _deduplicate(all_papers)
    _progress("Searched", f"Found {len(papers)} unique papers.", len(papers))
    return plan, all_papers, papers


def ai_screen_papers(
    client: anthropic.Anthropic,
    research_question: str,
    papers: list[dict],
) -> list[dict]:
    """Let Claude select the 15-25 most relevant papers from the candidate pool."""
    papers_summary = json.dumps(
        [{
            "i": i,
            "title": p["title"],
            "authors": p.get("authors", [])[:2],
            "year": p.get("year", ""),
            "abstract": p.get("abstract", "")[:300],
            "source": p.get("source", ""),
            "citations": p.get("citations", 0),
        } for i, p in enumerate(papers)],
        ensure_ascii=False,
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Research question: "{research_question}"

Select the 15-25 most relevant, high-quality papers from the list below.
Prioritise: high citation counts, peer-reviewed venues, direct relevance.

{papers_summary}

Return ONLY a JSON array of 0-based indices, e.g. [0, 3, 7, …]""",
        }],
    )
    try:
        indices = json.loads(resp.content[0].text)
        return [papers[i] for i in indices if i < len(papers)]
    except Exception:
        return papers[:20]


def synthesize_review(
    client: anthropic.Anthropic,
    research_question: str,
    selected_papers: list[dict],
    plan: dict,
    n_raw: int,
    n_deduped: int,
    citation_style: str = "APA",
    progress_callback=None,
) -> str:
    """
    Phase 3: write the full systematic review from selected papers.
    Handles continuation if the model hits the output token limit.
    """
    def _progress(phase, message, n=0):
        if progress_callback:
            progress_callback(LitReviewProgress(phase, message, n))

    _progress("Synthesizing", "Writing thematic synthesis…")

    selected_full = json.dumps(
        [{
            "index": i + 1,
            "title": p["title"],
            "authors": p.get("authors", []),
            "year": p.get("year", ""),
            "abstract": p.get("abstract", ""),
            "url": p.get("url", ""),
            "doi": p.get("doi", ""),
            "source": p.get("source", ""),
            "citations": p.get("citations", 0),
        } for i, p in enumerate(selected_papers)],
        ensure_ascii=False,
    )

    synthesis_prompt = f"""Write a comprehensive systematic literature review on: "{research_question}"

Citation style: {citation_style}

PICO framework:
{json.dumps(plan.get('pico', {}), indent=2)}

Inclusion criteria: {', '.join(plan.get('inclusion_criteria', []))}
Exclusion criteria: {', '.join(plan.get('exclusion_criteria', []))}

REAL SEARCH COUNTS — use these EXACT numbers in the Methods section:
- Records identified (before deduplication): {n_raw}
- Records after deduplication: {n_deduped}
- Records included after screening: {len(selected_papers)}

Papers available for citation (ONLY cite papers from this list):
{selected_full}

Rules:
- Cite inline as [1], [2], etc. using the index numbers above
- NEVER fabricate references not in the list
- The References section must list every paper you cite
- Do NOT include author disclosure or funding statements
- Stay tightly on the topic "{research_question}" — do not drift into a sub-niche
- Write thematically, not paper-by-paper"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=LITREVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    review_text = response.content[0].text

    # Continuation loop — up to 3 extra passes if the model hits the token limit
    cont = 0
    while response.stop_reason == "max_tokens" and cont < 3:
        cont += 1
        _progress("Synthesizing", f"Review is long — continuing (part {cont + 1})…")
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=LITREVIEW_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": synthesis_prompt},
                {"role": "assistant", "content": review_text},
                {"role": "user", "content":
                    "Continue the literature review from exactly where you left off. "
                    "Do NOT repeat any text already written. Continue seamlessly."},
            ],
        )
        review_text += response.content[0].text

    _progress("Complete", "Review generated!", len(selected_papers))
    return review_text


def verify_citations(selected_papers: list[dict]) -> list[dict]:
    """
    Check each paper's DOI via CrossRef.
    Returns a list of status dicts: {index, title, status, doi, url, journal, year}
    Status: "verified" | "preprint" | "not_found" | "no_doi"
    """
    results = []
    for i, p in enumerate(selected_papers):
        doi = p.get("doi", "").strip()
        url = p.get("url", "")
        source = p.get("source", "")

        entry = {
            "index": i + 1,
            "title": p.get("title", "")[:80],
            "doi": doi,
            "url": url,
            "status": "no_doi",
            "journal": "",
        }

        if doi:
            try:
                result = asyncio.run(verify_doi(doi))
                if result.get("valid"):
                    entry["status"] = "verified"
                    entry["journal"] = result.get("journal", "")
                else:
                    entry["status"] = "not_found"
            except Exception:
                entry["status"] = "not_found"
        elif "arxiv" in url.lower() or source in ("arXiv", "bioRxiv/medRxiv"):
            entry["status"] = "preprint"
        else:
            entry["status"] = "no_doi"

        results.append(entry)
    return results


def build_disclaimer(n_raw: int, n_deduped: int, n_included: int) -> str:
    return f"""

---

> **⚠️ AI-Generated Review — Transparency Notice**
> Generated by Claude claude-sonnet-4-6 with live searches of PubMed, Semantic Scholar, arXiv, and bioRxiv/medRxiv on {__import__('datetime').date.today()}.
> - **Real search counts:** {n_raw} papers retrieved → {n_deduped} after deduplication → {n_included} human/AI selected.
> - **Citation caution:** References were drawn from live database results. Verify independently before academic use.
> - This is a research aid, not a substitute for a human-conducted systematic review.
"""
