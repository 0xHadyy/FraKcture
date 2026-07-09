import streamlit as st

home_page = st.Page("./pages/home.py", title="FraKcture")
research_land = st.Page("./pages/research_landscape.py", title="Research Explorer")
topic_page = st.Page("./pages/topic_explore.py", title="Topic Explorer")

pg = st.navigation([home_page, research_land, topic_page])
pg.run()
