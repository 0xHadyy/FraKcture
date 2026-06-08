# flitring the 1.7M papers to ~ 50k (for my weekend project scope lol)
# Full data set Path
# Taking papers from the DL boom early 2013 till 2026
import json
import random
from pathlib import Path

# 554777 paper


class FilterPaper:
    def __init__(
        self,
        max_results: int = 1000,
        data_path: str = "",
        min_year: int = 2013,
        max_year: int = 2026,
    ):
        self.max_results = max_results
        self.data_path = data_path
        self.min_year = min_year
        self.max_year = max_year

    def _load_data(self):
        file_path = Path(self.data_path)

        with open(file_path, "r") as f:
            data = json.load(f)
            return data

    def _save_data(self, data):
        filename = f"Arxiv_sample_{int(str(self.max_results)[:2])}K.json"
        file_path = f"{Path.cwd()}/data/{filename}"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def prop_sample(self) -> tuple:
        self.prop = {}
        self.total = 0

        data = self._load_data()
        for d in data:
            year = int(d["year"])
            self.total += 1
            self.prop[year] = self.prop.get(year, 0) + 1

        print(self.prop)
        results = {}
        for year, count in self.prop.items():
            ratio = count / self.total
            results[year] = {"percentage": f"{ratio:.4}", "count": count}

        return (results, data)

    def stratification(self, sample_size):
        # take all the data , for example 2023 -> 12.06%, calculate 12.06% of 50k, fetch that number off 2013
        results, data = self.prop_sample()

        rng = random.Random(42)
        total_sample = 0
        papers_by_year = {}

        for paper in data:
            year = int(paper["year"])
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper)

        sampled_papers = []
        for year, stats in results.items():
            count = stats["count"]
            perc_sample = count / self.total

            sample_n = round(perc_sample * sample_size)

            year_paper = papers_by_year.get(year, [])
            print(year_paper)
            if sample_n > 0 and year_paper:
                sampled = rng.sample(year_paper, min(sample_n, len(year_paper)))
                sampled_papers.extend(sampled)
                # Sample randomly from that year
            total_sample += sample_n

        rng.shuffle(sampled_papers)
        self._save_data(sampled_papers)
        return sampled_papers

    def filter_paper(self, categories: list):
        papers = []

        dataset_path = "/home/hady/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/288/arxiv-metadata-oai-snapshot.json"

        json_path = Path(dataset_path)
        with open(json_path, "r") as f:
            i = 0
            for line in f:
                if i >= self.max_results:
                    break
                paper = json.loads(line)
                papers_cat = paper.get("categories", "").split()
                date = paper["versions"][0]["created"]
                year = 0
                date = date.split()
                try:
                    year = int(date[3])
                except ValueError:
                    pass

                if (
                    any(cat in papers_cat for cat in categories)
                    and year >= self.min_year
                    and year <= self.max_year
                ):
                    print(f"paper added year : {year}!")
                    i += 1
                    papers.append(
                        {
                            "id": paper.get("id"),
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract", ""),
                            "year": year,
                            "published": paper.get("versions", [{}])[0].get(
                                "created", ""
                            )
                            if paper.get("versions")
                            else "",
                            "categories": papers_cat,
                            "authors": paper.get("authors", ""),
                            "url": f"https://arxiv.org/pdf/{paper.get('id')}.pdf",
                        }
                    )
            return papers


data_path = "./data/ArXiv_2013_2026.json"

filter = FilterPaper(data_path=data_path, max_results=50000)
# result = filter.prop_sample()
result = filter.stratification(50000)
# pprint.pprint(result)
