# import tfidf from sklean
# produce the spare matrx
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = Path("./data/processed_data.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

X = None
vectorize = TfidfVectorizer(
    max_features=1000,
    min_df=1,
    max_df=0.85,
)
for d in data:
    text = d["processed_text"]

    X = vectorize.fit_transform(text)


df_tf_idf = pd.DataFrame(X.toarray(), columns=vectorize.get_feature_names_out())
print(df_tf_idf.round(3))
print(vectorize.get_feature_names_out())

# print(X)
