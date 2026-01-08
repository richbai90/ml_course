import streamlit as st

pg = st.navigation([st.Page("conv_1d.py"), st.Page("conv_2d.py")])
pg.run()
