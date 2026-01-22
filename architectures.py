import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Deep Learning Architectures")
lightbulb_callout("Big Picture", r'''An analysis of inductive biases in network design. We compare MLPs (dense connectivity), CNNs (translational invariance/locality), and Transformers (global attention mechanisms) to understand which architectures are
                  mathematically best suited for specific data modalities (e.g., Euclidean grids vs. sequences).''')
st.markdown("""
In Computational Imaging, we choose architectures based on the structure of the data and the problem we are solving (e.g., Classification, Tracking, or Domain Translation).
""")

# Shared Helper: Generate a dummy image
def get_dummy_image(size=64):
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    # A simple pattern: Circle + Wavy lines
    Z = np.exp(-(X**2 + Y**2)/2.0) + 0.5 * np.sin(3*X)
    return (Z - Z.min()) / (Z.max() - Z.min())

img_data = get_dummy_image()

# --- Architecture Selector ---
tab1, tab2, tab3 = st.tabs(["1. Feed Forward (MLP)", "2. CNN", "3. Transformer"])

# ==========================================
# TAB 1: FEED FORWARD (MLP)
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("The Multi-Layer Perceptron (MLP)")
        st.markdown("""
        **What is it?**
        A network where every neuron in one layer connects to every neuron in the next. It treats input as a flat vector, ignoring spatial structure (top-left is as "far" from top-right as it is from bottom-right).
        
        **Use Cases in Computational Imaging:**
        * **Coordinate-based Networks (NeRFs):** Instead of processing a whole image, we input coordinates $(x, y)$ and the MLP outputs the color $(r, g, b)$.
        * **Per-Pixel Operations:** Simple color space transformations.
        
        **Code Structure:**
        ```python
        nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        ```
        """)
    
    with col2:
        st.subheader("Demo: Implicit Representation")
        st.write("Visualizing an MLP 'painting' an image pixel by pixel.")
        
        # Simulate an MLP learning a coordinate mapping
        # We just color the image based on distance from center to simulate a learned field
        res = st.slider("Resolution (Grid Size)", 10, 50, 20)
        
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        xx, yy = np.meshgrid(x, y)
        
        # Simulating MLP Output: f(x,y) -> intensity
        # Ideally we'd run a real torch model, but for speed/demo we use a math function
        # representing a "learned" continuous function.
        zz = np.sin(np.sqrt(xx**2 + yy**2) * 10)
        
        fig = px.imshow(zz, color_continuous_scale='Viridis', title="MLP Learned Field (Coordinate -> Color)")
        fig.update_layout(xaxis_showticklabels=False, yaxis_showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: CNN
# ==========================================
with tab2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Convolutional Neural Networks (CNN)")
        st.markdown("""
        **What is it?**
        Instead of dense connections, we slide small filters (kernels) across the image. This preserves **Spatial Invariance** (a cat in the top-left is the same as a cat in the bottom-right).
        
        **Use Cases in Computational Imaging:**
        * **Image Classification:** Detecting objects.
        * **Image-to-Image Translation (U-Net):** Denoising, Deblurring, MRI Reconstruction.
        * **Tracking:** Extracting features from current frame and searching for them in the next.
        
        **Code Structure:**
        ```python
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        ```
        """)
        
    with col2:
        st.subheader("Demo: Feature Extraction")
        st.write("How a CNN 'sees' edges.")
        
        # Manual Convolution for Demo
        kernel_choice = st.selectbox("Select Filter", ["Vertical Edge", "Horizontal Edge", "Sharpen"])
        
        if kernel_choice == "Vertical Edge":
            k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif kernel_choice == "Horizontal Edge":
            k = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            
        # Apply filter (using scipy or simple correlation)
        from scipy.signal import convolve2d
        filtered_img = convolve2d(img_data, k, mode='same')
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(img_data, caption="Input Image", clamp=True, width=200)
        with col_b:
            st.image(filtered_img, caption=f"Feature Map ({kernel_choice})", clamp=True, width=200)

# ==========================================
# TAB 3: TRANSFORMER
# ==========================================
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Vision Transformers (ViT)")
        st.markdown("""
        **What is it?**
        Originally for text, Transformers use "Self-Attention" to relate every part of the input to every other part. For images, we chop the image into **Patches** (little squares), flatten them, and treat them like words in a sentence.
        
        **Use Cases in Computational Imaging:**
        * **Global Context:** Unlike CNNs (which look locally), Transformers see the whole image at once. Great for removing large obstructions or inpainting.
        * **Multi-Modal:** Connecting text to images (e.g., CLIP).
        
        **Code Structure:**
        ```python
        # Input: [Batch, Patches, Dim]
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        output = layer(input_patches)
        ```
        """)
        
    with col2:
        st.subheader("Demo: Patch Embeddings")
        st.write("How a Transformer breaks down an image.")
        
        patch_size = st.slider("Patch Size", 4, 16, 8, step=4)
        
        # Visualize Patches
        img_size = 64
        # Create a grid visualization
        
        fig = go.Figure()
        
        # Add the base image
        fig.add_trace(go.Heatmap(z=img_data, colorscale='gray', showscale=False))
        
        # Draw grid lines
        for i in range(0, img_size, patch_size):
            # Horizontal
            fig.add_shape(type="line", x0=0, y0=i, x1=img_size, y1=i, line=dict(color="red", width=1))
            # Vertical
            fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=img_size, line=dict(color="red", width=1))
            
        fig.update_layout(
            title=f"Splitting Image into {patch_size}x{patch_size} Tokens",
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            height=400,
            width=400
        )
        st.plotly_chart(fig)
        
        num_patches = (img_size // patch_size) ** 2
        st.info(f"The Transformer sees this image as a sequence of **{num_patches} vectors**, not a grid of pixels.") 

st.divider()

ask_ai()
