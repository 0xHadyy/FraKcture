import json
from datetime import datetime
from pathlib import Path

import arxiv


class ArxivClient:
    def __init__(self, max_results=80):
        self.client = arxiv.Client(page_size=20, delay_seconds=5, num_retries=5)
        self.max_results = max_results

    def fetch(self, query=None, categories=None):
        if query is None:
            query = " OR ".join([f"cat:{cat}" for cat in categories])

        fetch_query = query
        fetchs = arxiv.Search(
            query=fetch_query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []

        for result in self.client.results(fetchs):
            papers.append(
                {
                    "id": result.get_short_id(),
                    "arxiv_id": result.entry_id,
                    "title": result.title.strip(),
                    "abstract": result.summary.strip(),
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                }
            )

        return papers

    def save_papers(self, papers):
        filename = f"ArXiv_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = f"{Path.cwd()}/data/{filename}"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2)

        return filepath

    def load_json(self, filepath):
        filepath = Path(filepath)

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data
