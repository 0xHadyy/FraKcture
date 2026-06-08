import json
import re
from pathlib import Path

import spacy

nlp = spacy.load("en_core_web_sm")


class Preprocess:
    def __init__(
        self,
        lemmatization=False,
    ):
        self.lemmatization = lemmatization
        self.pattern = re.compile(r"\W+")
        self.custom_stop_words = {
            "paper",
            "work",
            "study",
            "research",
            "approach",
            "method",
            "methods",
            "proposed",
            "proposal",
            "result",
            "results",
            "experimental",
            "experiment",
            "experiments",
            "analysis",
            "evaluate",
            "evaluation",
            "performance",
            "show",
            "shows",
            "demonstrate",
            "demonstrates",
            "demonstrated",
            "novel",
            "state-of-the-art",
            "sota",
            "based",
            "using",
            "used",
            "use",
            "framework",
            "model",
            "models",
            "technique",
            "techniques",
            "algorithm",
            "algorithms",
            "system",
            "systems",
            "problem",
            "problems",
            "task",
            "tasks",
            "application",
            "applications",
        }

    def _load_json(self, filepath):
        filepath = Path(filepath)

        with filepath.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except FileNotFoundError:
                raise FileNotFoundError(f"File with path {filepath} couldn't be loaded")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid Json: {e} ")

    def _clean_text(self, text):
        """Remove LaTeX and extra whitespace"""
        # Remove LaTeX math mode
        text = re.sub(r"\$[^$]+\$", "", text)  # Inline math
        text = re.sub(r"\$\$[^$]+\$\$", "", text)  # Display math
        text = re.sub(r"\\[a-zA-Z]+", "", text)  # LaTeX commands
        text = re.sub(r"\{[^}]*\}", "", text)  # LaTeX braces
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process(self, papers):
        # remove latex
        # remove common words
        #
        path = "./data/ArXiv_20260605.json"
        data = self._load_json(path)

        # data =nlp(text.lower)
        processed_data = []
        for d in data:
            processed_tokens = []
            text = f"{d['title']} {d['abstract']}"
            text = self._clean_text(text)
            text = nlp(text.lower())

            for token in text:
                if (
                    not token.is_stop
                    and token.text not in self.custom_stop_words
                    and not token.is_punct
                    and not token.is_space
                    and not token.is_digit
                    and len(token.text) > 2
                ):
                    if self.lemmatization:
                        token_text = token.lemma_
                    else:
                        token_text = token.text
                    processed_tokens.append(token_text)

            # for date in full_date:
            # print(f"this is the date{date}")
            # try:
            # int_date = int(date)
            # except ValueError:
            #    int_date = 0
            # if int_date > 1000:
            #    pass
            #    break

            processed_paper = {
                "id": d["id"],
                "year": d["year"],
                "title": d["title"],
                "abstract": d["abstract"],
                "processed_text": processed_tokens,
                "processed_text_joined": " ".join(processed_tokens),
                "processed_text_length": len(processed_tokens),
                "original_text_length": len(text),
            }
            processed_data.append(processed_paper)
        return processed_data


process = Preprocess(lemmatization=True)
data = process.process(papers=None)

filepath = f"{Path.cwd()}/data/processed_data.json"

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
