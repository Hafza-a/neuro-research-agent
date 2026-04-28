"""
Peer agent — searches for papers the student might have missed, then writes collegial feedback.
Has tool use (PubMed, Semantic Scholar, arXiv). Returns both text feedback and any new papers found.
"""
import asyncio
import json
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from agent.tools.arxiv_search import search_arxiv
from prompts.peer_system import PEER_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for neuroscience papers on the review topic.",
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
        "description": "Search Semantic Scholar for highly-cited or recent papers on the topic.",
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


def peer_review_and_search(
    client: anthropic.Anthropic,
    draft: str,
    research_question: str,
    current_papers: list,
    on_tool_call=None,
) -> tuple:
    """
    Peer searches for missing papers and writes feedback.

    Returns:
        feedback_text (str): collegial feedback markdown
        new_papers (list[dict]): papers found via search not already in the pool
    """
    existing_titles = {(p.get("title") or "").lower().strip() for p in current_papers}
    new_papers: list = []

    # Build a compact list of already-covered paper titles
    covered = "\n".join(
        f"- {p.get('title', 'Unknown')[:90]} ({p.get('year', '')})"
        for p in current_papers[:40]
    )

    # Truncate draft to avoid huge context (peer sees enough to understand scope)
    draft_preview = draft[:5000] + ("\n\n[... draft continues ...]" if len(draft) > 5000 else "")

    loop_messages = [{
        "role": "user",
        "content": f"""Your peer is writing a literature review on:
**"{research_question}"**

They have already included these papers (do NOT suggest these):
{covered}

Here is their current draft — read it carefully to understand what they have covered:
---
{draft_preview}
---

Now search for papers they might have missed, then write your collegial feedback following your instructions.""",
    }]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            system=PEER_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=loop_messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            # Final text response — extract feedback
            feedback = "".join(b.text for b in response.content if b.type == "text")
            return feedback, new_papers

        # Notify callback of tool calls (for UI display)
        if on_tool_call:
            for tu in tool_uses:
                on_tool_call(tu.name, tu.input)

        loop_messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            papers_found = asyncio.run(_dispatch(tu.name, tu.input))
            result_str = json.dumps(papers_found, ensure_ascii=False)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })
            # Collect papers not already in the pool
            if isinstance(papers_found, list):
                for p in papers_found:
                    if isinstance(p, dict) and p.get("title"):
                        title_key = p["title"].lower().strip()
                        if title_key not in existing_titles:
                            new_papers.append(p)
                            existing_titles.add(title_key)

        loop_messages.append({"role": "user", "content": tool_results})
