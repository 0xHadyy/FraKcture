import joblib
import numpy as np
import pandas as pd

from pipeline.cluster import Cluster


class Artifact:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.embeddings = np.load("./data/lsa_embaddings_norm.npy")
        self.cluster_instance = Cluster()
        self.km = joblib.load("./models/kmeans.joblib")
        self.df = None
        self.close_title = []

    def paper_topics_artifact(self):

        d = self.km.transform(self.embeddings)
        self.df = pd.read_json(self.dataset_path)
        self.df["cluster"] = self.km.labels_
        self.df["distance"] = np.min(d, axis=1)

        self.df.to_parquet("./data/papers_with_topics.parquet", index=False)

    def topics_metadata_artifact(self):
        clusters = self.cluster_instance.map_cluster(km=self.km)
        topic_df = pd.DataFrame(clusters)

        topic_df.to_csv("./data/topic_metadata.csv", index=False)

        return topic_df

    def closest_paper_artifact(self):
        if self.df is None:
            self.paper_topics_artifact()
            print("done")

        for label in np.unique(self.km.labels_):
            cluster = self.df[self.df.cluster == label]

            closest = cluster.nsmallest(10, "distance")
            self.close_title.append(
                {"cluster": label, "close_papers": closest["title"].tolist()}
            )

        topic_paper_distance = pd.DataFrame(self.close_title)
        topic_paper_distance.to_csv("./data/topic_paper_distance.csv")

        return topic_paper_distance

    def topic_evo_artifact(self):

        topic_year = (
            self.df.groupby(["year", "cluster"]).size().reset_index(name="paper_count")
        )

        topic_year.to_parquet("./data/topic_year_count.parquet")
        return topic_year


data_path = "./data/processed_data_50k_5.json"
artifact = Artifact(dataset_path=data_path)

r = artifact.closest_paper_artifact()
y = artifact.topic_evo_artifact()
