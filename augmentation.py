import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

st.set_page_config(layout="wide")
st.title("Lab: Data Augmentation")

st.markdown("""
**Why augment data?** In computational imaging, our models often need to be **invariant** to certain changes. 
* A cell viewed under a microscope is the same cell whether it's rotated 90° or 180°.
* A sensor reading might have varying levels of thermal noise.

If we don't have enough training data covering these variations, we **simulate** them during training. This prevents the model from memorizing exact pixel values and forces it to learn robust features.
""")

# --- Helper: Dummy Data Generator (Microscopy style) ---
@st.cache_data
def get_microscopy_sample(size=256):
    """Generates a synthetic fluorescence-like image."""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Create "cells" (blobs)
    Z = np.zeros_like(X)
    centers = [(0,0), (1.5, 1.5), (-1.2, 0.8), (0.5, -2.0)]
    for x0, y0 in centers:
        dist = np.sqrt((X - x0)**2 + (Y - y0)**2)
        Z += np.exp(-dist**2 / 0.4)  # Gaussian blobs
    
    # Normalize to 0-1
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z.astype(np.float32)

# --- 1. Data Loading ---
col_cfg, col_view = st.columns([1, 2])

with col_cfg:
    st.subheader("1. Input Data")
    data_source = st.radio("Choose Source:", ["Synthetic Microscopy", "Upload Your Own"])
    
    img_tensor = None
    
    if data_source == "Synthetic Microscopy":
        raw_data = get_microscopy_sample()
        # Convert to Tensor [C, H, W]
        img_tensor = torch.tensor(raw_data).unsqueeze(0) 
        st.caption("Generated a synthetic fluorescence intensity map.")
        
    else:
        uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L") # Convert to grayscale for simplicity
            transform = transforms.ToTensor()
            img_tensor = transform(image)
        else:
            st.warning("Please upload an image to proceed.")
            st.stop()

# --- 2. Augmentation Pipeline ---
st.divider()

col_controls, col_display = st.columns([1, 2])

with col_controls:
    st.subheader("2. Define Transforms")
    
    st.markdown("**Geometric Transforms**")
    st.caption("Changes the spatial arrangement of pixels.")
    
    angle = st.slider("Rotation (Degrees)", -180, 180, 0)
    do_h_flip = st.checkbox("Horizontal Flip")
    do_v_flip = st.checkbox("Vertical Flip")
    scale_zoom = st.slider("Zoom (Center Crop scale)", 0.5, 1.5, 1.0)
    
    st.markdown("---")
    st.markdown("**Photometric (Physics) Transforms**")
    st.caption("Simulates sensor imperfections or lighting changes.")
    
    noise_level = st.slider("Gaussian Noise (Sigma)", 0.0, 0.2, 0.0, step=0.01)
    brightness = st.slider("Brightness Factor", 0.5, 2.0, 1.0)
    blur_sigma = st.slider("Blur (Defocus)", 0.0, 3.0, 0.0)

# --- 3. Apply Logic ---
# Note: In a real training loop, we use 'transforms.Compose' with random probabilities.
# Here, we apply them deterministically to visualize the effect.

augmented = img_tensor.clone()

# Apply Geometry
augmented = F.rotate(augmented, angle)
if do_h_flip:
    augmented = F.hflip(augmented)
if do_v_flip:
    augmented = F.vflip(augmented)
if scale_zoom != 1.0:
    _, h, w = augmented.shape
    new_h, new_w = int(h / scale_zoom), int(w / scale_zoom)
    augmented = F.center_crop(augmented, [new_h, new_w])
    augmented = F.resize(augmented, [h, w]) # Resize back to original for display

# Apply Photometrics
augmented = F.adjust_brightness(augmented, brightness)

if blur_sigma > 0:
    # Kernel size must be odd
    k_size = int(blur_sigma * 4) + 1 
    if k_size % 2 == 0: k_size += 1
    augmented = F.gaussian_blur(augmented, kernel_size=k_size, sigma=blur_sigma)

if noise_level > 0:
    noise = torch.randn_like(augmented) * noise_level
    augmented = augmented + noise
    augmented = torch.clamp(augmented, 0, 1)

# --- 4. Visualization ---
with col_display:
    st.subheader("3. Result Comparison")
    
    # Convert back to numpy for Plotly
    # [C, H, W] -> [H, W] (assuming grayscale)
    img_orig_np = img_tensor.squeeze().numpy()
    img_aug_np = augmented.squeeze().numpy()
    
    # Side-by-side Plot
    fig = go.Figure()
    
    # Use subplots? Or just st.columns inside
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Original Input**")
        fig1 = px.imshow(img_orig_np, color_continuous_scale='gray', title="Original")
        fig1.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.write("**Augmented Batch**")
        fig2 = px.imshow(img_aug_np, color_continuous_scale='gray', title="Transformed")
        fig2.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# --- Code Snippet ---
st.markdown("### How to implement this in PyTorch")
st.code(f"""
from torchvision import transforms

# The training transform pipeline
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees={abs(angle) if angle != 0 else 15}),
    {'transforms.RandomHorizontalFlip(p=0.5),' if do_h_flip else '# transforms.RandomHorizontalFlip(p=0.5),'}
    {'transforms.RandomVerticalFlip(p=0.5),' if do_v_flip else '# transforms.RandomVerticalFlip(p=0.5),'}
    transforms.ColorJitter(brightness={brightness}),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, {blur_sigma if blur_sigma > 0 else 2.0})),
    transforms.ToTensor()
])

# Note: Pytorch doesn't have a built-in "AddGaussianNoise" in standard transforms,
# so we often write a custom Lambda function for that.
""", language="python")
