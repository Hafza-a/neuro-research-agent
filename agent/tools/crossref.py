import httpx

BASE = "https://api.crossref.org"


async def verify_doi(doi: str) -> dict:
    """Return metadata for a DOI, or an error dict if not found."""
    doi = doi.strip().lstrip("https://doi.org/").lstrip("http://dx.doi.org/")
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(
                f"{BASE}/works/{doi}",
                headers={"User-Agent": "NeuroResearchAgent/1.0 (mailto:research@example.com)"},
            )
            resp.raise_for_status()
            msg = resp.json().get("message", {})
            authors_raw = msg.get("author", [])
            authors = [
                f"{a.get('family', '')} {a.get('given', '')[:1]}".strip()
                for a in authors_raw
            ]
            date_parts = msg.get("published", {}).get("date-parts", [[""]])[0]
            year = str(date_parts[0]) if date_parts else ""
            return {
                "valid": True,
                "title": (msg.get("title") or [""])[0],
                "authors": authors,
                "year": year,
                "journal": (msg.get("container-title") or [""])[0],
                "doi": doi,
                "url": f"https://doi.org/{doi}",
            }
        except Exception as e:
            return {"valid": False, "doi": doi, "error": str(e)}
