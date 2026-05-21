"""
Adversary agent — adversarial peer reviewer that actively hunts for weaknesses.
Has tool use to search for missing or contradicting papers.
Returns (critique: str, score: int).
"""
import asyncio
import json
import re
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from prompts.adversary_system import ADVERSARY_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for papers that may challenge or be missing from the draft.",
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
        "description": "Search Semantic Scholar for highly-cited papers that should be in any review on this topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 8},
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
        return []
    except Exception:
        return []


def _parse_score(text: str) -> int:
    match = re.search(r"VERDICT:\s*(\d+)/10", text, re.IGNORECASE)
    if match:
        return max(1, min(10, int(match.group(1))))
    lower = text.lower()
    if "accept" in lower and "revision" not in lower and "reject" not in lower:
        return 9
    if "minor revision" in lower:
        return 7
    if "major revision" in lower:
        return 5
    if "reject" in lower:
        return 3
    return 6


def adversary_critique(
    client: anthropic.Anthropic,
    draft: str,
    research_question: str,
    current_papers: list,
    on_tool_call=None,
    usage_out: dict = None,
) -> tuple:
    """
    Adversarially reviews the draft, searches for missing/contradicting papers.

    Returns:
        critique (str):    full structured critique markdown
        score (int):       1–10
        new_papers (list): papers found by adversary not in current pool
    """
    existing_titles = {(p.get("title") or "").lower().strip() for p in current_papers}
    new_papers: list = []
    total_input = 0
    total_output = 0

    loop_messages = [{
        "role": "user",
        "content": (
            f"Review this literature review on: \"{research_question}\"\n\n"
            "Search for missing or contradicting papers, then write your adversarial critique "
            "following your instructions.\n\n"
            f"---\n{draft}\n---"
        ),
    }]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            system=ADVERSARY_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=loop_messages,
        )
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            critique = "".join(b.text for b in response.content if b.type == "text")
            score = _parse_score(critique)
            if usage_out is not None:
                usage_out["input_tokens"] = total_input
                usage_out["output_tokens"] = total_output
            return critique, score, new_papers

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
