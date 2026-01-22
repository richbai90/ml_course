import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Lab: Live Inference Playground")
lightbulb_callout('Big Picture', r'''The transition from training graphs to inference graphs. This covers the strict protocol required for deployment: disabling gradient computation (no_grad), freezing batch normalization statistics (eval mode), and
                  handling tensor batching for deterministic forward propagation.''')

st.markdown("""
**Training is only half the battle.** To use a model in the real world (Inference), we must strictly follow the pipeline the model was trained with.

**Key Differences from Training:**
1.  **No Gradients:** We use `torch.no_grad()` to save memory.
2.  **Eval Mode:** We call `model.eval()` to freeze layers like Dropout and Batch Normalization.
3.  **Batching:** Even for a single image, the model expects a batch dimension: `[1, Channels, Height, Width]`.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Use a lightweight modern architecture
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval() # CRITICAL: Switch to inference mode
    
    # Get the automatic transforms defined by the creators of the weights
    transform = weights.transforms()
    
    # Get class labels (ImageNet classes)
    classes = weights.meta["categories"]
    return model, transform, classes

try:
    with st.spinner("Loading Model (MobileNetV3)..."):
        model, transform, classes = load_model()
except Exception as e:
    st.error(f"Error loading model. Make sure you have internet access to download weights.\n\n{e}")
    st.stop()

# --- 2. INPUT ---
col_input, col_process, col_output = st.columns([1, 1, 1])

with col_input:
    st.subheader("1. Input Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Your Upload", use_container_width=True)
    else:
        # Default placeholder if nothing uploaded
        st.info("Upload an image to start!")
        st.stop()

# --- 3. PRE-PROCESSING ---
with col_process:
    st.subheader("2. What the Model Sees")
    
    # A. Apply Transforms (Resize -> CenterCrop -> Normalize)
    tensor_image = transform(input_image)
    
    # B. Add Batch Dimension [C, H, W] -> [1, C, H, W]
    batch_tensor = tensor_image.unsqueeze(0)
    
    st.markdown(f"""
    **Tensor Shape:** `{list(batch_tensor.shape)}`
    
    The model cannot "see" colors like we do. It sees normalized float values (usually between -2 and 2).
    """)
    
    # Visualize the normalized tensor (Channel 0)
    # We un-normalize roughly for display or just show raw heatmap
    raw_layer = tensor_image[0].numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=raw_layer, 
        colorscale='Viridis',
        showscale=False
    ))
    fig.update_layout(
        title="Normalized Feature Map (Channel R)", 
        xaxis_showticklabels=False, 
        yaxis_showticklabels=False,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This is the mathematical representation of the Red channel after normalization.")

# --- 4. INFERENCE & OUTPUT ---
with col_output:
    st.subheader("3. Prediction")
    
    if st.button("Run Inference", type="primary"):
        with torch.no_grad():
            # Forward Pass
            output = model(batch_tensor)
            
            # Convert raw logits to probabilities (Softmax)
            probabilities = F.softmax(output, dim=1)[0]
            
            # Get Top 5
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
        # Display Results
        st.write("### Top 5 Predictions")
        
        for i in range(5):
            score = top5_prob[i].item()
            class_name = classes[top5_catid[i].item()]
            
            # Visual Bar
            st.markdown(f"**{class_name}**")
            st.progress(score)
            st.caption(f"Confidence: {score:.2%}")

# --- 5. CODE SNIPPET ---
st.divider()
st.subheader("Implementation Code")
st.code("""
# The Inference Ritual
input_image = Image.open("test.jpg")

# 1. Transform
tensor = transform(input_image)

# 2. Add Batch Dimension (The most common error!)
batch = tensor.unsqueeze(0) 

# 3. Inference
model.eval() # Freeze batchnorm/dropout
with torch.no_grad():
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    
print(probs.topk(5))
""", language="python")

st.divider()
ask_ai()
