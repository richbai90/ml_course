import streamlit as st

pg = st.navigation([
    st.Page("transform.py"),
    st.Page("hidden_layers.py"),
    st.Page("gradient_descent.py"),
    st.Page("conv_1d.py"), 
    st.Page("conv_2d.py")
])
pg.run()
