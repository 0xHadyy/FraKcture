import json
import re
from pathlib import Path

import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class Preprocess:
    def __init__(self, lemmatization=False, data_path=""):
        self.lemmatization = lemmatization
        self.data_path = data_path
        self.pattern = re.compile(r"\W+")
        self.custom_stop_words = {
            # Original arXiv academic words
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
            "propose",
            "introduce",
            "provide",
            "present",
            "compare",
            "achieve",
            "improve",
            "address",
            "include",
            "enable",
            "available",
            "different",
            "new",
            "exist",
            "require",
            "perform",
            "outperform",
            "challenge",
            "datum",
            "time",
            "real",
            "design",
            "art",
            "level",
            "process",
            "set",
            "quality",
            "low",
            "space",
            "world",
            "user",
            "input",
            "enhance",
            "fine",
            "strategy",
            "end",
            "target",
            "point",
            "specific",
            "test",
            "reduce",
            "make",
            "made",
            "making",
            "get",
            "getting",
            "got",
            "see",
            "seeing",
            "seen",
            "look",
            "looking",
            "looked",
            "find",
            "finding",
            "found",
            "type",
            "types",
            "way",
            "ways",
            "part",
            "parts",
            "set",
            "sets",
            "number",
            "numbers",
            "value",
            "values",
            "form",
            "forms",
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

    def process(self):
        # remove latex
        # remove common words
        #
        path = Path(self.data_path)
        data = self._load_json(path)
        i = 0
        # data =nlp(text.lower)
        texts = []
        for paper in data:
            text = f"{paper['title']} {paper['abstract']}"
            text = self._clean_text(text)
            texts.append(text.lower())
        processed_data = []
        docs = nlp.pipe(texts, batch_size=200, n_process=4)
        for d, doc in zip(data, docs):
            processed_tokens = []

            for token in doc:
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

            processed_paper = {
                "id": d["id"],
                "year": d["year"],
                "title": d["title"],
                "abstract": d["abstract"],
                "processed_text": processed_tokens,
                "processed_text_joined": " ".join(processed_tokens),
                "processed_text_length": len(processed_tokens),
                "original_text_length": len(d["title"] + " " + d["abstract"]),
            }
            i += 1
            print(f"{i} paper processed")
            processed_data.append(processed_paper)
        return processed_data


dataset_path = "./data/Arxiv_sample_50K.json"

process = Preprocess(lemmatization=True, data_path=dataset_path)
data = process.process()

filepath = f"{Path.cwd()}/data/processed_data_50k_2.json"

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Preprocessing is done...")
