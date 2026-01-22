import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import signal

from utils import lightbulb_callout

# Wide mode is essential for this three-plot layout
st.set_page_config(layout="wide", page_title="2D Convolution: Raster Lab")

st.title("🖼️ 2D Convolution: Raster Scan & Spatial Features")
lightbulb_callout("Big Picture", r'''Extending convolution to 2D spatial domains. This demonstrates the Raster Scan operation central to Computer Vision. You will analyze how discrete kernels (like Sobel operators) exploit spatial invariance to compute
                  gradients and extract morphological features like edges and textures.''')
st.markdown("---")


# --- 1. Setup Input Image ---
img_size = 30
image = np.zeros((img_size, img_size))
# Create high-contrast features: a square and an inner gray block
image[5:25, 5:25] = 1.0
image[12:18, 12:18] = 0.5

# --- 2. Sidebar Configuration ---
with st.sidebar:
    st.header("Kernel & Scan Settings")
    kernel_choice = st.selectbox(
        "Kernel Function (h)", ["Edge Detection (Sobel)", "Blur (Average)", "Sharpen"]
    )
    stride = st.select_slider("Stride (S)", options=[1, 2, 3, 5])

# Kernel Definition Logic
if kernel_choice == "Edge Detection (Sobel)":
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
elif kernel_choice == "Blur (Average)":
    kernel = np.ones((5, 5)) / 25
else:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# --- 3. Raster Calculations ---
rows = np.arange(0, img_size, stride)
cols = np.arange(0, img_size, stride)
total_steps = len(rows) * len(cols)

# Centralized Slider for Raster Scan
step_idx = st.select_slider(
    "Raster Scan Step (Time Line)", options=range(total_steps), value=0
)

# Map linear index to 2D coordinates
out_row_idx = step_idx // len(cols)
out_col_idx = step_idx % len(cols)
y_pos, x_pos = rows[out_row_idx], cols[out_col_idx]

# Perform full convolution and take absolute value for visualization consistency
full_conv_abs = np.abs(signal.convolve2d(image, kernel, mode="same"))
stride_result = full_conv_abs[::stride, ::stride]

# --- 4. Main Visualization Row ---
left_col, mid_col, right_col = st.columns([1, 1, 1])

with left_col:
    st.subheader(f"1. Input Image Scan")
    fig_img = go.Figure(data=go.Heatmap(z=image, colorscale="gray", showscale=False))

    # Red Box highlighting the current kernel window
    k_h, k_w = kernel.shape
    fig_img.add_shape(
        type="rect",
        x0=x_pos - k_w // 2 - 0.5,
        y0=y_pos - k_h // 2 - 0.5,
        x1=x_pos + k_w // 2 + 0.5,
        y1=y_pos + k_h // 2 + 0.5,
        line=dict(color="Red", width=3),
    )
    fig_img.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_img, use_container_width=True)
    st.caption(f"Kernel center currently at: ({x_pos}, {y_pos})")

with mid_col:
    st.subheader("2. Kernel Weights (h)")
    # Visualization of the kernel weights themselves
    fig_k = go.Figure(data=go.Heatmap(z=kernel, colorscale="RdBu", zmid=0))
    fig_k.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_k, use_container_width=True)
    st.caption(f"Currently using: {kernel_choice}")

with right_col:
    st.subheader("3. Feature Map (Absolute Value)")

    # Linear reveal logic to match the raster scan
    flat_result = stride_result.flatten()
    display_flat = np.zeros_like(flat_result)
    display_flat[: step_idx + 1] = flat_result[: step_idx + 1]
    display_result = display_flat.reshape(stride_result.shape)

    vmax = np.max(stride_result) if np.max(stride_result) > 0 else 1
    fig_res = go.Figure(
        data=go.Heatmap(z=display_result, colorscale="Viridis", zmin=0, zmax=vmax)
    )

    # White cursor in output to match red box in input
    fig_res.add_shape(
        type="rect",
        x0=out_col_idx - 0.5,
        y0=out_row_idx - 0.5,
        x1=out_col_idx + 0.5,
        y1=out_row_idx + 0.5,
        line=dict(color="White", width=2),
    )

    fig_res.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_res, use_container_width=True)
    st.caption(f"Output Dimensions: {stride_result.shape[1]}x{stride_result.shape[0]}")

# --- 5. Full-Width Explanation Row ---
st.markdown("---")
st.header("Understanding the 2D Raster Convolution")

st.markdown(
    """
### The Raster Scan Pattern
In this demonstration, the kernel doesn't move arbitrarily. It follows a **Raster Scan**—the same way a CRT monitor or a standard image sensor reads pixels: left-to-right, then top-to-bottom. 



### Why Absolute Value?
For kernels like **Sobel (Edge Detection)**, the mathematical result contains both positive and negative values. Positive values indicate a transition from dark to light, while negative values indicate light to dark. 
* By plotting the **Absolute Value** ($|x * h|$), we visualize the "strength" of an edge regardless of its direction. 
* This is a standard step in Computer Vision, ensuring that edges appear as bright features on a dark background.

### Stride and Dimensionality
Notice that as you increase the **Stride**, the output feature map (Plot 3) physically shrinks. This represents a form of downsampling. In deep learning, high strides are used to reduce the computational load and increase the "receptive field" of the network, allowing it to see larger structures with fewer pixels.
"""
)
