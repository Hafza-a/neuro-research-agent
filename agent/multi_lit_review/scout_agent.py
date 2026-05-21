"""
Scout agent — enriches the paper pool before writing begins.
Runs additional targeted searches, then produces a field briefing
that gives the Writer thematic orientation before drafting.

Returns: (briefing: str, new_papers: list[dict])
"""
import asyncio
import json
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from agent.tools.arxiv_search import search_arxiv
from prompts.scout_system import SCOUT_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for neuroscience papers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_semantic_scholar",
        "description": "Search Semantic Scholar — best for finding highly-cited foundational papers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_arxiv",
        "description": "Search arXiv for preprints and computational neuroscience work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


async def _dispatch(name: str, inputs: dict) -> list:
    try:
        if name == "search_pubmed":
            return await search_pubmed(inputs["query"], inputs.get("max_results", 8))
        elif name == "search_semantic_scholar":
            return await search_semantic_scholar(inputs["query"], inputs.get("max_results", 8))
        elif name == "search_arxiv":
            return await search_arxiv(inputs["query"], inputs.get("max_results", 5))
        return []
    except Exception:
        return []


def _format_pool_titles(papers: list) -> str:
    return "\n".join(
        f"  [{i+1}] {p.get('title', 'Unknown')[:90]} ({p.get('year', 'n.d.')}) — {p.get('source', '')}"
        for i, p in enumerate(papers[:50])
    )


def scout_enrich(
    client: anthropic.Anthropic,
    research_question: str,
    papers: list,
    on_tool_call=None,
    usage_out: dict = None,
) -> tuple:
    """
    Scout enriches the paper pool and produces a field briefing.

    Returns:
        briefing (str):     thematic field briefing for the Writer
        new_papers (list):  additional papers found via search (not in original pool)
    """
    existing_titles = {(p.get("title") or "").lower().strip() for p in papers}
    new_papers: list = []
    total_input = 0
    total_output = 0

    loop_messages = [{
        "role": "user",
        "content": (
            f"Research question: \"{research_question}\"\n\n"
            f"The paper pool already contains {len(papers)} papers:\n"
            f"{_format_pool_titles(papers)}\n\n"
            "Your tasks:\n"
            "1. Run 3–5 targeted searches to find critical missing papers "
            "(foundational work, recent advances, meta-analyses, alternative perspectives).\n"
            "2. After searching, write the complete Field Briefing following your instructions.\n"
            "Focus searches on what is most likely missing from the pool above."
        ),
    }]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            system=SCOUT_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=loop_messages,
        )
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            briefing = "".join(b.text for b in response.content if b.type == "text")
            if usage_out is not None:
                usage_out["input_tokens"] = total_input
                usage_out["output_tokens"] = total_output
            return briefing, new_papers

        if on_tool_call:
            for tu in tool_uses:
                on_tool_call(tu.name, tu.input)

        loop_messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            found = asyncio.run(_dispatch(tu.name, tu.input))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(found, ensure_ascii=False),
            })
            for p in found:
                if isinstance(p, dict) and p.get("title"):
                    key = p["title"].lower().strip()
                    if key not in existing_titles:
                        new_papers.append(p)
                        existing_titles.add(key)

        loop_messages.append({"role": "user", "content": tool_results})
