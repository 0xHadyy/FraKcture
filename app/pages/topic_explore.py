import ast

import pandas as pd
import plotly.express as px
import streamlit as st

st.markdown(
    """
    <style>
        header.stAppHeader {
            background-color: transparent;
        }
        section.stMain .block-container {
            padding-top: 1rem; /* Adjust this value (e.g., 0rem for no space) */
            z-index: 1;
        }
    </style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
    <h1 style='font-size: 4rem; 
               font-weight: 700; margin: 0; text-align:center;'>
        Fra<span style='color:#B31B1B'>Kc</span>ture Research Topics
    </h1>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    topics_metadata = pd.read_csv("./data/topics_metadata_mod.csv")
    topic_distance = pd.read_csv("./data/topic_paper_distance.csv")
    papers = pd.read_parquet("./data/papers_with_topics.parquet")

    df = pd.read_parquet("./data/df_umap.parquet")
    return topics_metadata, topic_distance, papers, df


topics_metadata, topic_dist, papers, df = load_data()
# st.write(topics_metadata)
# st.write(topic_dist)
# st.write(papers)
# st.write(df)


st.markdown("## Select Topic to Explore")

selected_topic = st.selectbox(
    "Research Topic", sorted(topics_metadata["labels"].unique())
)


col1, col2 = st.columns(2, border=True)
with col1:
    st.markdown("### Keywords")
    raw_keywords = topics_metadata.loc[
        topics_metadata["labels"] == selected_topic, "keywords"
    ].iloc[0]

    keywords = ast.literal_eval(raw_keywords)

    cols = st.columns(8)

    for i, keyword in enumerate(keywords):
        with cols[i % 8]:
            st.badge(keyword)
with col2:
    col12, col22 = st.columns(2)

    with col12:
        num_paper = df[df["topic"] == selected_topic]
        st.metric("Paper Number", len(num_paper))
    with col22:
        low_year = num_paper["year"].min()
        peak_year = num_paper["year"].mode()[0]
        st.metric("First Appear", f"{low_year}")
        st.metric("Peak Year", f"{peak_year}")


trend = num_paper.groupby("year").size().reset_index(name="papers")
peak = trend.loc[trend["papers"].idxmax()]
fig = px.line(
    trend,
    x="year",
    y="papers",
    markers=True,
)

fig.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=30, b=10),
)
fig.add_vline(
    x=peak["year"],
    line_dash="dash",
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("## Representative Papers")
st.markdown(
    "The papers that are the closest to the topic & define the esense of that field"
)
selected_topic_cluster = num_paper["cluster"].iloc[0]
raw_rep_papers = topic_dist.loc[
    topic_dist["cluster"] == selected_topic_cluster, "close_papers"
].iloc[0]

rep_papers = ast.literal_eval(raw_rep_papers)


for title in rep_papers:
    row = papers.loc[papers["title"] == title].iloc[0]

    with st.container(border=True):
        st.markdown(f"### {row['title']}")

        st.caption(f"{row['year']}")

        st.write(row["abstract"][:250] + "...")

        st.link_button("Open on arXiv", f"https://arxiv.org/abs/{row['id']}")
