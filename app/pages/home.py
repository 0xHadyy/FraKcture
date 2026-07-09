import pandas as pd
import streamlit as st

# from utils.loader import load_papers
st.set_page_config(page_title="FraKcture", page_icon="🎸", layout="wide")


@st.cache_data
def load_df():
    df = pd.read_parquet("./data/papers_with_topics.parquet")
    return df


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
# st.write("# FraKcture", alignment="center")

st.markdown(
    """
    <h1 style='font-size: 4rem; 
               font-weight: 700; margin: 0; text-align:center;'>
        Fra<span style='color:#B31B1B'>Kc</span>ture
    </h1>
""",
    unsafe_allow_html=True,
)
# st.header("Case Study for Machine Leanrning Research Papers Evolution (2013-2026).",text_alignment="center",)

st.write("### Discipline")
st.write(
    "FraKcture, inspired of **King Crimson's** Fracture & **Discpline** album structure, is an interactive semantic map of AI and machine learning research spanning the deep learning era (2013–2026). Discover the latent research communities and enables exploration of how new ideas emerge, evolve, and spread throughout the machine learning ecosystem. This project is an ongoing study, with additional analyses and features planned in future releases."
)
st.write("")
st.write("## Dataset Info")
col1, col2, col3, col4 = st.columns(4)

col1.metric(label="ML Total Papers", value="~500,000")
col2.metric(label="Papers Analysed", value="50,000")
col3.metric(label="Years", value="2013-2026")
col4.metric(label="Discoverd", value="30 Topics")

st.write("---")
st.write("")
st.write("## Data Flow Diagram")
st.write("")
st.write("")
with st.container(horizontal_alignment="center"):
    st.image("./assets/diagram_light.png")

st.write("")
st.html(
    "<p style='text-align:center;'>For FraKcture 0.1v the main NLP technique used to discover the <strong>hidden</strong> topics in the research papers is <strong>Latent Semantic Analysis</strong> which is a classical NLP method but proves it's effectivness given the results of this study.</p>"
)
df = load_df()

st_df = pd.DataFrame(df["year"].value_counts(), df["year"].unique())
st.bar_chart(st_df, x_label="years", y_label="Papers", stack=False, color="#B31B1B")


st.caption(
    "Distribution of machine learning papers in the sampled arXiv corpus (2013–2026). "
    "The sample preserves the temporal distribution(Stratisfied) of the original dataset.",
    text_alignment="center",
)
st.write("")
st.write("")
st.write("")
st.write("")

st.markdown(
    "<h1 style='text-align: center;'>Explore Fra<span style='color:#B31B1B';>Kc</span>ture</h1>",
    unsafe_allow_html=True,
)

st.write("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Research Landscape")
    st.write("""
        - Explore the latent space through an interactive **UMAP** visualization 
        """)
with col2:
    st.subheader("Topic Explorer")
    st.write("""
        - **Inspect research areas top papers** : by distance , year , relevance..
        - **Discover the important keywords & related fields** 
        """)
with col3:
    st.subheader("Topic Evolution")
    st.write(
        """- Analyze how AI/ML research evolved over the years and how new areas emerge."""
    )
