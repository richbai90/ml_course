import uuid

import streamlit as st


def lightbulb_callout(title, content):
    # Generate a unique ID for this specific box so we can target it with JS
    # Inject CSS to turn the blue st.info box into your yellow style
    st.markdown("""
    <style>
    div[data-testid="stAlert"] {
        background-color: #fff9e6;
        border: none;
        border-left: 5px solid #ffc107;
        color: #333;
    }
    div[data-testid="stAlert"] > div {
        color: #333; /* Text color */
    }
    /* Hide standard icon if you want to replace it, or keep it */
    </style>
    """, unsafe_allow_html=True)
    
    # Use native component -> Latex works automatically!
    st.info(f"**{title}**\n\n{content}", icon="💡")
