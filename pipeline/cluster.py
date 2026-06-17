import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Cluster:
    def __init__(self, n_clusters=30, top_n=10):
        self.tf_idf = joblib.load("./models/tf_idf_vectorizer.joblib")
        self.svd = joblib.load("./models/truncated_svd.joblib")
        self.n_clusters = n_clusters
        self.km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.silh_score = 0
        self.centriods = []
        self.top_n = top_n
        self.clusters = []
        self.word_centroids = []

    def get_feature_names(self):
        return self.tf_idf.get_feature_names_out()

    def cluster(self):
        embedding = np.load("./data/lsa_embaddings_norm.npy")
        labels = self.km.fit_predict(embedding)
        self.silh_score = silhouette_score(embedding, labels)
        self.centriods = self.km.cluster_centers_
        return self.centriods, self.silh_score

    def map_cluster(self):
        self.word_centroids = self.centriods @ self.svd.components_

        for i in range(self.n_clusters):
            top_indices = np.argsort(self.word_centroids[i])[-self.top_n :][::-1]
            terms = self.tf_idf.get_feature_names_out()
            top_terms = [terms[j] for j in top_indices]
            self.clusters.append(top_terms)

        return self.clusters


# documents -> DTM(tf-idf, feature extraction) -> Truncated SVD -> 50 important component -> cluster


km = Cluster()

km.cluster()
clusters = km.map_cluster()
i = 1
for cluster in clusters:
    print(f"Cluster{i}\n")

    print(cluster)
    i += 1
