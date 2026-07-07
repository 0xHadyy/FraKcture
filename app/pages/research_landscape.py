import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FraKcture Landscape", page_icon="🎸", layout="wide")


@st.cache_data
def load_df():

    df_umap = pd.read_parquet("./data/df_umap.parquet")
    return df_umap


df = load_df()


@st.cache_data
def fig(df=None):
    fig = px.scatter(
        df, x="x", y="y", color="topic", hover_data=["title", "year", "distance"]
    )
    return fig


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
      Research Landscape 
    </h1>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Filters")
    query = st.text_input(
        label="", placeholder="Search Paper", label_visibility="visible"
    )

    years = st.slider("Publication Year", 2013, 2026, (2013, 2026))
    distance = st.slider("Maximum centriod distance", 0.0, 1.0, (0.0, 1.0))

    selected_topics = st.multiselect(
        "Topics",
        sorted(df["topic"].unique()),
        default=sorted(df["topic"].unique()),
    )


def filter_df(df, years, selected_topics, distance, query):

    filter_df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]

    filter_df = filter_df[filter_df["topic"].isin(selected_topics)]
    filter_df = filter_df[
        (filter_df["distance"] >= float(distance[0]))
        & (filter_df["distance"] <= float(distance[1]))
    ]

    if query:
        filter_df = filter_df[
            filter_df["title"].str.contains(query, case=False, na=False)
        ]

    return filter_df


filtered_df = filter_df(df, years, selected_topics, distance, query)
top_topic = (
    filtered_df["topic"].value_counts().idxmax() if not filtered_df.empty else "—"
)
peak_year = (
    filtered_df["year"].value_counts().idxmax() if not filtered_df.empty else "—"
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Visible Papers", len(filtered_df))
with col3:
    st.metric("Topics", f"{filtered_df['topic'].nunique()} Topic")
with col2:
    st.metric("Years", f"{years[0]}-{years[1]}")


fig = fig(df=filtered_df)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(
    height=700,
    margin=dict(l=10, r=10, t=40, b=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.update_traces(
    marker=dict(
        size=4,
        opacity=0.65,
    )
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Each point is a paper projected to 2D via UMAP from 50-dimensional LSA embeddings. "
    "Proximity indicates semantic similarity. Cluster labels are derived from K-Means centroids (k=30).",
    text_alignment="center",
)
