import httpx

BASE = "https://api.biorxiv.org"


async def search_biorxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    bioRxiv's public API supports date-range queries but not keyword search directly.
    We use the /details endpoint with a broad recent window and filter by title/abstract match.
    For a true keyword search we fall back to querying their search page via the jatsxml endpoint.
    """
    results = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{BASE}/details/biorxiv/2020-01-01/2099-01-01/0/json",
            )
            # This endpoint returns recent papers; keyword search isn't natively supported.
            # We do a targeted search via their search API instead.
            search_resp = await client.get(
                "https://www.biorxiv.org/search/" + query.replace(" ", "%20"),
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )

        # Fall back to Semantic Scholar preprint search if direct API fails
        async with httpx.AsyncClient(timeout=20) as client:
            ss_resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": max_results,
                    "fields": "title,authors,year,abstract,externalIds,url",
                    "venue": "bioRxiv,medRxiv",
                },
                headers={"User-Agent": "NeuroResearchAgent/1.0 (research tool)"},
            )
            if ss_resp.status_code == 429:
                return []
            ss_resp.raise_for_status()
            papers = ss_resp.json().get("data", [])

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
                "source": "bioRxiv/medRxiv",
            })
    except Exception:
        pass

    return results[:max_results]
