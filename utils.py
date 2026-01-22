import inspect
import uuid

import streamlit as st


def ask_ai():
    """
    Adds an expander with a copyable prompt containing the current page's source code.
    """
    # 1. Get the source code of the file that called this function
    # stack()[1] refers to the caller (the page script)
    caller_frame = inspect.stack()[1]
    source_file_path = caller_frame.filename
    
    with open(source_file_path, "r") as f:
        source_code = f.read()

    # 2. Construct the Prompt
    ai_prompt = f"""Please help me better understand the concepts described in this module. I have attached the source code for the Streamlit page I am looking at. Please explain the mathematical concepts in a way that a graduate student would understand. Ignore streamlit specific details.

---
SOURCE CODE:
{source_code}
"""

    # 3. Render the Button/Output
    # We use an expander to keep the UI clean
    with st.expander("🤖 Ask AI about this page"):
        st.caption("Click the copy button in the top-right of the box below, then paste into your AI assistant.")
        st.code(ai_prompt, language="text")

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
