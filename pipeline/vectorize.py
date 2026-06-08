# import tfidf from sklean
# produce the spare matrx
import json
from collections import Counter
from pathlib import Path
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

counter = Counter()

file_path = Path("./data/processed_data_50k_2.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
all_text = []
for d in data:
    text = d["processed_text_joined"]
    # counter.update(d["processed_text"])
    all_text.append(text)

X = None
for min_df in [5, 10, 20, 50]:
    vectorize = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.85,
    )
    X = vectorize.fit_transform(all_text)
    pprint(X.shape)

# pprint(counter.most_common(100))


# feature_names = vectorize.get_feature_names_out()

# print(X)
