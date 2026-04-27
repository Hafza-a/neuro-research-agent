import httpx
import xmltodict

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            search_resp = await client.get(
                f"{BASE}/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            )
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            fetch_resp = await client.get(
                f"{BASE}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            )
            fetch_resp.raise_for_status()
        except Exception:
            return []
        data = xmltodict.parse(fetch_resp.text)

    articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
    if isinstance(articles, dict):
        articles = [articles]

    results = []
    for art in articles:
        try:
            citation = art["MedlineCitation"]
            article = citation["Article"]
            title = article.get("ArticleTitle", "")
            if isinstance(title, dict):
                title = title.get("#text", str(title))

            abstract_obj = article.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_obj, list):
                abstract = " ".join(
                    (a.get("#text", str(a)) if isinstance(a, dict) else str(a))
                    for a in abstract_obj
                )
            elif isinstance(abstract_obj, dict):
                abstract = abstract_obj.get("#text", "")
            else:
                abstract = str(abstract_obj) if abstract_obj else ""

            authors_raw = article.get("AuthorList", {}).get("Author", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = [
                f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                for a in authors_raw
                if isinstance(a, dict) and a.get("LastName")
            ]

            pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4] if pub_date.get("MedlineDate") else "")

            pmid = str(citation.get("PMID", {}).get("#text", citation.get("PMID", "")))
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract[:800],
                "url": url,
                "doi": "",
                "source": "PubMed",
            })
        except Exception:
            continue

    return results
