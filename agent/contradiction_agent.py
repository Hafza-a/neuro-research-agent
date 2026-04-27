import asyncio
import json
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from agent.tools.arxiv_search import search_arxiv
from agent.tools.crossref import verify_doi
from prompts.contradiction_system import CONTRADICTION_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

# Re-use the same tool definitions as the research agent
TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for peer-reviewed neuroscience literature supporting or contradicting a claim.",
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
        "description": "Search Semantic Scholar — best for finding highly-cited replication or meta-analysis papers.",
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
        "description": "Search arXiv for preprints and computational studies.",
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
        "name": "verify_doi",
        "description": "Verify a DOI and get full citation metadata.",
        "input_schema": {
            "type": "object",
            "properties": {"doi": {"type": "string"}},
            "required": ["doi"],
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
        elif name == "verify_doi":
            r = await verify_doi(inputs["doi"])
        else:
            r = {"error": f"Unknown tool: {name}"}
        return json.dumps(r, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def detect_contradictions(
    client: anthropic.Anthropic,
    claim: str,
    on_tool_call=None,
) -> str:
    """
    Search the literature for evidence supporting and contradicting a claim,
    then produce a structured Contradiction Report.
    """
    loop_messages = [{
        "role": "user",
        "content": f"""Investigate this neuroscience claim for supporting and contradicting evidence:

"{claim}"

Search PubMed and Semantic Scholar for:
1. Papers that support this claim (including the original finding if known)
2. Papers that contradict, fail to replicate, or show opposite results
3. Any meta-analyses or systematic reviews on this topic

Then write the full Contradiction Report following your instructions.""",
    }]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=6000,
            system=CONTRADICTION_SYSTEM_PROMPT,
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
