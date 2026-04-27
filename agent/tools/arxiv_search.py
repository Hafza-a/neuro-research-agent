import httpx
import xmltodict

BASE = "https://export.arxiv.org/api/query"


async def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            resp = await client.get(
                BASE,
                params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
            )
            resp.raise_for_status()
        except Exception:
            return []
        data = xmltodict.parse(resp.text)

    feed = data.get("feed", {})
    entries = feed.get("entry", [])
    if isinstance(entries, dict):
        entries = [entries]

    results = []
    for e in entries:
        authors_raw = e.get("author", [])
        if isinstance(authors_raw, dict):
            authors_raw = [authors_raw]
        authors = [a.get("name", "") for a in authors_raw if isinstance(a, dict)]

        published = e.get("published", "")
        year = published[:4] if published else ""

        arxiv_id = e.get("id", "")
        doi_link = ""
        for link in (e.get("link", []) if isinstance(e.get("link"), list) else [e.get("link", {})]):
            if isinstance(link, dict) and link.get("@title") == "doi":
                doi_link = link.get("@href", "")

        summary = e.get("summary", "").replace("\n", " ").strip()

        results.append({
            "title": e.get("title", "").replace("\n", " ").strip(),
            "authors": authors,
            "year": year,
            "abstract": summary[:800],
            "url": arxiv_id,
            "doi": doi_link,
            "source": "arXiv",
        })
    return results
