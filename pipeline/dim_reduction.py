import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import umap


# reducer = umap.UMAP(n_neighbors=100, min_dist=0.01, n_components=2)
# a = pd.read_parquet("./data/papers_with_topics.parquet")
class DimReduction:
    def __init__(self, num_neighbors=100, min_dist=0.01):
        self.num_neighbors = num_neighbors
        self.min_dist = min_dist
        self.embeddings = np.load("./data/lsa_embaddings_norm.npy")
        self.km = joblib.load("./models/kmeans.joblib")
        self.meta_data = pd.read_csv("./data/topics_metadata_mod.csv")
        self.mapped_labels = np.load("./data/name_label_cluster.npy", allow_pickle=True)
        self.X_umap = None
        self.umap_df = None

    def reduce(self):
        reducer = umap.UMAP(n_neighbors=self.num_neighbors, min_dist=self.min_dist)
        self.X_umap = reducer.fit_transform(self.embeddings)

        # saving umap embeddings
        np.save("./data/X_umap_embeddings_test.npy", self.X_umap)

    def umap_artifact(self):
        papers = pd.read_parquet("./data/papers_with_topics.parquet")
        self.umap_df = pd.DataFrame(
            {
                "x": self.X_umap[:, 0],
                "y": self.X_umap[:, 1],
                "cluster": self.km.labels_,
                "title": papers["title"],
                "year": papers["year"],
                "topic": self.mapped_labels,
            }
        )
        self.umap_df.to_parquet("./data/df_umap.parquet", index=False)

    def plot(self):

        if self.umap_df is None:
            self.umap_df = pd.read_parquet("./data/df_umap.parquet")

        fig = px.scatter(
            self.umap_df, x="x", y="y", color="topic", hover_data=["title", "year"]
        )
        fig.show()


dim = DimReduction()

dim.reduce()
dim.umap_artifact()
dim.plot()

# new_label = np.array([])
# for label in km.labels_:
#    new_label = np.append(new_label, meta_data[meta_data.cluster == label]["labels"])

# umap_df = pd.DataFrame(
#    {
#        "x": X_umap[:, 0],
#        "y": X_umap[:, 1],
#        "cluster": km.labels_,
#        "title": a["title"],
#        "year": a["year"],
#        "topic": new_label,
#    }
# )

# umap_df.to_parquet("./data/umap_embedings.parquet", index=False)

# umap_df = pd.read_parquet("./data/umap_embedings.parquet")
# fig = px.scatter(umap_df, x="x", y="y", color="topic", hover_data=["title", "year"])

# fig.show()


# print(umap_df)
# print(f"labels of km are type of {type(km.labels_)}")

# print(km.labels_.shape)
# print("---------------")

# np.save("./data/name_label_cluster.npy", new_label)
# print("done")
# print("plotitng...\n")
# palette = sns.color_palette("", 30)
# umap.plot.points(X_umap, labels=labels_named, color_key=custom_colors)
# plt.show()


# np.save("./data/umap_embeddings.npy", X_umap)


# print(a["cluster"].value_counts().sort_index())

# for clust in range(30):
#    df2 = df[df["cluster"] == clust]
#    print(f"the cluster label is {df2['labels']}\n")
#    print(f"{a[a['cluster'] == clust].nsmallest(5, 'distance')['title']}\n")
#    print("----------------------------------------------------------------------")
# print(a)
