# flitring the 1.7M papers to ~ 10k-8k (for my weekend project scope lol)


# Full data set Path
import json
from pathlib import Path

from arxiv_fetch import ArxivClient


def filter_papers(
    categories=["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"], max_results=10000
):
    papers = []
    dataset_path = "/home/hady/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/288/arxiv-metadata-oai-snapshot.json"

    json_path = Path(dataset_path)

    with open(json_path, "r") as f:
        i = 0
        for line in f:
            if i >= max_results:
                print(f"the value of i is : {i}")
                break

            paper = json.loads(line)

            papers_cat = paper.get("categories", "").split()

            if any(cat in papers_cat for cat in categories):
                i += 1
                papers.append(
                    {
                        "id": paper.get("id"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract", ""),
                        "published": paper.get("versions", [{}])[0].get("created", "")
                        if paper.get("versions")
                        else "",
                        "categories": papers_cat,
                        "authors": paper.get("authors", ""),
                        "url": f"https://arxiv.org/pdf/{paper.get('id')}.pdf",
                    }
                )

        return papers


papers = filter_papers()

print(f"Got {len(papers)}, Time to json dump them....")
client = ArxivClient()

client.save_papers(papers)
print(f"Got {len(papers)}, all papers are ready to be processed...")
