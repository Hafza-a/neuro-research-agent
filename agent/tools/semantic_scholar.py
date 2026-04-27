import httpx

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,abstract,externalIds,citationCount,url"


async def search_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                f"{BASE}/paper/search",
                params={"query": query, "limit": max_results, "fields": FIELDS},
                headers={"User-Agent": "NeuroResearchAgent/1.0 (research tool)"},
            )
            if resp.status_code == 429:
                return [{"title": "", "error": "Semantic Scholar rate limit — try again in a moment", "source": "Semantic Scholar"}]
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            return []
        papers = resp.json().get("data", [])

    results = []
    for p in papers:
        authors = [a.get("name", "") for a in p.get("authors", [])]
        doi = p.get("externalIds", {}).get("DOI", "")
        results.append({
            "title": p.get("title", ""),
            "authors": authors,
            "year": str(p.get("year", "")),
            "abstract": (p.get("abstract") or "")[:800],
            "url": p.get("url", ""),
            "doi": doi,
            "citations": p.get("citationCount", 0),
            "source": "Semantic Scholar",
        })
    return results
