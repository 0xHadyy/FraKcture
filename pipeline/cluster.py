import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_values = [8, 10, 15, 17, 20]


tf_idf = joblib.load("./models/tf_idf_vectorizer.joblib")
svd = joblib.load("./models/truncated_svd.joblib")

terms = tf_idf.get_feature_names_out()
km = KMeans(n_clusters=14, random_state=42, n_init=10)
embedding = np.load("./data/lsa_embaddings.npy")
labels = km.fit_predict(embedding)
print(f"the labels are :{labels}")
silh_score = silhouette_score(embedding, labels)
print(f"the silh score for k={14} is {silh_score}")

print(f"the labels are -> {km.labels_}")


centroids = km.cluster_centers_
print(centroids)


word_centroids = centroids @ svd.components_


top_n = 15

for i in range(14):
    top_indices = np.argsort(word_centroids[i])[-top_n:][::-1]
    top_terms = [terms[j] for j in top_indices]

    print(f"\nCluster {i}")
    print(top_terms)
