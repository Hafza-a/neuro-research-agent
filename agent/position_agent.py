import asyncio
import json
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from agent.tools.arxiv_search import search_arxiv
from agent.tools.biorxiv import search_biorxiv
from prompts.position_system import POSITION_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for papers similar to the user's abstract.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_semantic_scholar",
        "description": "Search Semantic Scholar — best for citation counts and highly-cited related work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_arxiv",
        "description": "Search arXiv for preprints on the same topic.",
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
        "name": "search_biorxiv",
        "description": "Search bioRxiv/medRxiv for recent preprints on the same topic.",
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


async def _dispatch(name: str, inputs: dict) -> str:
    try:
        if name == "search_pubmed":
            r = await search_pubmed(inputs["query"], inputs.get("max_results", 10))
        elif name == "search_semantic_scholar":
            r = await search_semantic_scholar(inputs["query"], inputs.get("max_results", 10))
        elif name == "search_arxiv":
            r = await search_arxiv(inputs["query"], inputs.get("max_results", 8))
        elif name == "search_biorxiv":
            r = await search_biorxiv(inputs["query"], inputs.get("max_results", 5))
        else:
            r = {"error": f"Unknown tool: {name}"}
        return json.dumps(r, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def position_paper(
    client: anthropic.Anthropic,
    abstract: str,
    on_tool_call=None,
) -> str:
    """
    Given a paper abstract, search the literature for the most similar work
    and produce a full Positioning Report.
    """
    loop_messages = [{
        "role": "user",
        "content": f"""Analyse the following paper abstract and produce a full Positioning Report.

First search for the most similar existing published work using multiple database queries.
Search for: the core topic/method, key terms from the abstract, and potential competitor papers.

--- ABSTRACT ---
{abstract}
--- END ABSTRACT ---

Now produce the complete Positioning Report following your instructions.""",
    }]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=6000,
            system=POSITION_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=loop_messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            return "\n".join(b.text for b in response.content if b.type == "text")

        if on_tool_call:
            for tu in tool_uses:
                on_tool_call(tu.name, tu.input)

        loop_messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            result = asyncio.run(_dispatch(tu.name, tu.input))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })
        loop_messages.append({"role": "user", "content": tool_results})
