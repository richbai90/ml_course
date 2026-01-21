import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.vq import kmeans, vq

st.set_page_config(layout="wide")
st.title("Supervised vs. Unsupervised Learning")

st.markdown(r"""
In Computational Imaging, the choice of method depends on the data we have available:
* **Supervised:** We have an **Input** and a corresponding **Target (Ground Truth)**. We teach the model to map $x \to y$.
* **Unsupervised:** We only have **Inputs**. We ask the model to find structure or patterns within the data itself.
""")

tab1, tab2 = st.tabs(["1. Supervised Learning", "2. Unsupervised Learning"])

# --- TAB 1: SUPERVISED ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Mapping Input to Target")
        st.info("**Goal:** Minimize the error between the Model's Output and the Target (Ground Truth).")
        st.markdown(r"""
        **Imaging Example: Denoising**
        * **Input ($x$):** The Noisy Measurement.
        * **Target ($y$):** The Clean Image (Ground Truth).
        * **Loss:** $\text{Distance}(y, \hat{y})$
        
        **Imaging Example: MRI Reconstruction**
        * **Input ($x$):** Under-sampled K-space frequencies.
        * **Target ($y$):** Fully sampled MRI scan.
        
        The model learns the transformation function $f(x) \approx y$.
        """)
        
    with col2:
        st.subheader("Demo: Fitting a Target Function")
        # Generate Ground Truth
        x = np.linspace(0, 10, 100)
        target_y = np.sin(x)
        
        # Generate Input Data (Noisy observations of the target)
        noise_level = st.slider("Noise Level (Measurement Error)", 0.1, 1.0, 0.3)
        np.random.seed(42)
        input_y = target_y + np.random.normal(0, noise_level, len(x))
        
        fig = go.Figure()
        
        # 1. The Input
        fig.add_trace(go.Scatter(
            x=x, y=input_y, 
            mode='markers', 
            name='Input Data (x)',
            marker=dict(color='gray', opacity=0.6)
        ))
        
        # 2. The Target
        fig.add_trace(go.Scatter(
            x=x, y=target_y, 
            mode='lines', 
            name='Target / Ground Truth (y)',
            line=dict(color='green', width=4)
        ))
        
        # 3. Untrained Model
        untrained_model = np.sin(x) * 0.5 + 0.2
        show_untrained = st.checkbox("Show Untrained Model Prediction")
        if show_untrained:
            fig.add_trace(go.Scatter(
                x=x, y=untrained_model,
                mode='lines',
                name='Model Output (y_hat)',
                line=dict(color='red', dash='dot')
            ))
            
        fig.update_layout(
            title="Supervised: Learning the Mapping from Input to Target", 
            xaxis_title="Domain (e.g., Time/Space)",
            yaxis_title="Signal Intensity",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: UNSUPERVISED ---
with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Discovering Structure")
        st.warning(r"**Goal:** Group similar inputs without a Target.")
        st.markdown("""
        **Imaging Example: Segmentation**
        * **Input:** An image.
        * **Target:** None.
        
        We ask the model: *"Group these pixels into 3 categories based on their features."* The model might group sky pixels together and grass pixels together simply because they are statistically distinct, not because it knows what "sky" is.
        """)
        
    with col2:
        st.subheader("Demo: K-Means Clustering")
        
        # Generate Random Blobs
        n_points = 300
        b1 = np.random.normal(loc=[2, 2], scale=0.8, size=(n_points//3, 2))
        b2 = np.random.normal(loc=[6, 6], scale=1.0, size=(n_points//3, 2))
        b3 = np.random.normal(loc=[2, 8], scale=0.6, size=(n_points//3, 2))
        
        data = np.vstack([b1, b2, b3])
        
        k = st.slider("Number of Clusters (K)", 1, 6, 3)
        
        # Run K-Means
        centroids, distortion = kmeans(data, k)
        idx, _ = vq(data, centroids)
        
        fig_u = go.Figure()
        
        # Plot points colored by their assigned cluster
        fig_u.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode='markers',
            marker=dict(color=idx, colorscale='Rainbow', size=8, line=dict(width=1, color='DarkSlateGrey')),
            name='Input Data'
        ))
        
        # Plot Centroids
        fig_u.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(color='black', symbol='x', size=15),
            name='Cluster Centers'
        ))
        
        fig_u.update_layout(
            title=f"Unsupervised: The model found {k} groups automatically.",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            height=400
        )
        st.plotly_chart(fig_u, use_container_width=True) 
