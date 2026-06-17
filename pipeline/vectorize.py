from collections import Counter

import joblib
import numpy as np
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from pipeline.preprocess import Preprocess


class Vectorize:
    def __init__(
        self,
        min_df: int = 5,
        max_df: float = 0.85,
        ngram_range: tuple = (1, 1),
        data_path: str = " ",
    ):
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

        self.full_text = []
        self.X = None
        self.load_data = Preprocess()
        self.data_path = data_path
        self.data_cache = None
        self.feature_names = None

    def _load_data_once(self):
        if self.data_cache is None:
            self.data_cache = self.load_data._load_json(filepath=self.data_path)
        return self.data_cache

    def _most_common(self, common_num: int):
        counter = Counter()
        data = self._load_data_once()

        for d in data:
            counter.update(d["processed_text"])

        return counter.most_common(common_num)

    def _append_text(self):
        data = self._load_data_once()
        for d in data:
            # join the full strings(text+title) for each paper
            text = d["processed_text_joined"]
            self.full_text.append(text)

        return self.full_text

    # tfidf need full text , not tokens
    def vectorizer(self) -> tuple[csr_matrix, ndarray]:
        if not self.full_text:
            self._append_text()

        vectorize = TfidfVectorizer(
            min_df=self.min_df, max_df=self.max_df, ngram_range=self.ngram_range
        )
        # document matrix (N_docs x terms) sparse
        self.X = vectorize.fit_transform(self.full_text)
        sparse.save_npz("./data/tf_idf.npz", self.X)
        joblib.dump(vectorize, "./models/tf_idf_vectorizer.joblib")
        self.feature_names = vectorize.get_feature_names_out()
        return self.X, self.feature_names

    # Use the Singular Value decomp on the sparse TF-IDF Matrix
    def lsa(self) -> tuple[ndarray, float]:
        svd = TruncatedSVD(n_components=50, random_state=42)
        X_lsa = svd.fit_transform(self.X)
        X_lsa_norm = normalize(X_lsa)
        joblib.dump(svd, "./models/truncated_svd.joblib")
        np.save("./data/lsa_embaddings.npy", X_lsa)
        np.save("./data/lsa_embaddings_norm.npy", X_lsa_norm)

        for i in range(20):
            comp = svd.components_[i]
            top_idx = comp.argsort()[-10:][::-1]
            print(f"\n component {i}:")
            for idx in top_idx:
                print(self.feature_names[idx])
        return X_lsa, svd.explained_variance_ratio_.sum()


file_path = "./data/processed_data_50k_5.json"

vectorizer = Vectorize(data_path=file_path)
vectorizer._load_data_once()
full_text = vectorizer._append_text()
X, feature_names = vectorizer.vectorizer()
x_lsa, explained_var = vectorizer.lsa()
print(feature_names)
for feature in feature_names:
    pass

print(X.shape)
print("---------")

print(x_lsa.shape)
print(f"the explained variance ---> :{explained_var}")


# feature_names = vectorize.get_feature_names_out()
