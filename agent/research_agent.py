import asyncio
import json
import anthropic
from agent.tools.pubmed import search_pubmed
from agent.tools.semantic_scholar import search_semantic_scholar
from agent.tools.arxiv_search import search_arxiv
from agent.tools.biorxiv import search_biorxiv
from agent.tools.crossref import verify_doi
from prompts.research_system import RESEARCH_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for peer-reviewed neuroscience and biomedical literature. Returns title, authors, year, abstract, and URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for PubMed"},
                "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_semantic_scholar",
        "description": "Search Semantic Scholar for academic papers with citation counts. Best for finding highly-cited foundational work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_arxiv",
        "description": "Search arXiv for preprints in neuroscience, computational neuroscience, and related fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for arXiv"},
                "max_results": {"type": "integer", "default": 10, "description": "Maximum number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_biorxiv",
        "description": "Search bioRxiv and medRxiv for neuroscience preprints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for bioRxiv/medRxiv"},
                "max_results": {"type": "integer", "default": 5, "description": "Maximum number of results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "verify_doi",
        "description": "Verify a DOI and retrieve full citation metadata from CrossRef.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI string to verify (e.g. 10.1038/nature12345)"},
            },
            "required": ["doi"],
        },
    },
]


async def _dispatch_tool(name: str, inputs: dict) -> str:
    try:
        if name == "search_pubmed":
            result = await search_pubmed(inputs["query"], inputs.get("max_results", 10))
        elif name == "search_semantic_scholar":
            result = await search_semantic_scholar(inputs["query"], inputs.get("max_results", 10))
        elif name == "search_arxiv":
            result = await search_arxiv(inputs["query"], inputs.get("max_results", 10))
        elif name == "search_biorxiv":
            result = await search_biorxiv(inputs["query"], inputs.get("max_results", 5))
        elif name == "verify_doi":
            result = await verify_doi(inputs["doi"])
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_research_turn(client: anthropic.Anthropic, messages: list, on_tool_call=None) -> str:
    """
    Run one agentic turn of the research agent.
    Handles the tool-use loop synchronously (Streamlit friendly).
    Returns the final assistant text response.
    """
    loop_messages = list(messages)

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=RESEARCH_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=loop_messages,
        )

        # Collect tool calls and text from this response
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # If no tool calls, we have the final answer
        if not tool_uses:
            return "\n".join(b.text for b in text_blocks)

        # Notify UI about tool calls
        if on_tool_call:
            for tu in tool_uses:
                on_tool_call(tu.name, tu.input)

        # Add assistant message with tool use blocks
        loop_messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls
        tool_results = []
        for tu in tool_uses:
            result_str = asyncio.run(_dispatch_tool(tu.name, tu.input))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })

        loop_messages.append({"role": "user", "content": tool_results})

        # If stop_reason is end_turn after appending results, loop again for synthesis
        if response.stop_reason == "end_turn":
            # Shouldn't happen here since tool_uses were present, but be safe
            final = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=RESEARCH_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=loop_messages,
            )
            return "\n".join(b.text for b in final.content if b.type == "text")
