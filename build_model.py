import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Lab: Build Your Own Model")
lightbulb_callout('Big Picture', r'''Tensor algebra and dimensionality arithmetic. This tool assists in designing valid computational graphs by calculating tensor shapes through successive convolutional and linear transformations, ensuring mathematical
                  compatibility between layers.''')
st.markdown("""
**The Architect's Workbench.** Design a network layer-by-layer. This tool automatically calculates the **Output Shapes** to help you avoid the dreaded `RuntimeError: size mismatch`.
""")

# --- 1. GLOBAL SETTINGS ---
with st.sidebar:
    st.header("Input Data Dimensions")
    in_h = st.number_input("Height", 28, 256, 28, step=4)
    in_w = st.number_input("Width", 28, 256, 28, step=4)
    in_c = st.number_input("Channels (1=Gray, 3=RGB)", 1, 3, 1)
    
    st.header("Task Settings")
    num_classes = st.number_input("Output Classes", 2, 1000, 10)
    
    st.divider()
    model_type = st.radio("Architecture Type", ["Convolutional (CNN)", "Multi-Layer Perceptron (MLP/SLP)"])

# --- 2. ARCHITECTURE BUILDER ---

# Container for the generated code string
code_lines = []
dims_log = [] # To store shape changes: (Layer Name, Output Shape)

if model_type == "Convolutional (CNN)":
    st.subheader("CNN Builder")
    
    col_layers, col_viz = st.columns([1, 1])
    
    with col_layers:
        num_conv_layers = st.slider("Number of Conv Blocks", 1, 5, 2)
        
        layers = []
        
        # Track current dimensions for calculation
        current_c = in_c
        current_h = in_h
        current_w = in_w
        
        dummy_input = torch.randn(1, in_c, in_h, in_w)
        dims_log.append(("Input", f"{list(dummy_input.shape)}"))
        
        code_lines.append("self.features = nn.Sequential(")
        
        # --- DYNAMIC LAYER GENERATION ---
        for i in range(num_conv_layers):
            st.markdown(f"**Block {i+1}**")
            c1, c2, c3 = st.columns(3)
            
            out_channels = c1.selectbox(f"L{i+1} Filters", [8, 16, 32, 64, 128], index=i, key=f"f{i}")
            k_size = c2.selectbox(f"L{i+1} Kernel", [3, 5, 7], index=0, key=f"k{i}")
            stride = c3.selectbox(f"L{i+1} Stride", [1, 2], index=0, key=f"s{i}")
            
            # Add Conv Layer
            layers.append(nn.Conv2d(current_c, out_channels, k_size, stride))
            code_lines.append(f"    nn.Conv2d({current_c}, {out_channels}, kernel_size={k_size}, stride={stride}),")
            code_lines.append("    nn.ReLU(),")
            
            # Update Dims (Math check)
            current_h = int((current_h - k_size)/stride + 1)
            current_w = int((current_w - k_size)/stride + 1)
            current_c = out_channels
            
            dims_log.append((f"Conv2d_{i+1}", f"[1, {current_c}, {current_h}, {current_w}]"))
            
            # Optional Pooling
            use_pool = st.checkbox(f"Add MaxPool to Block {i+1}?", value=(i==0), key=f"p{i}")
            if use_pool:
                layers.append(nn.MaxPool2d(2, 2))
                code_lines.append("    nn.MaxPool2d(kernel_size=2, stride=2),")
                current_h //= 2
                current_w //= 2
                dims_log.append((f"MaxPool_{i+1}", f"[1, {current_c}, {current_h}, {current_w}]"))
            
            st.divider()

        code_lines.append(")")
        
        # --- FLATTEN & CLASSIFIER ---
        flat_size = current_c * current_h * current_w
        
        dims_log.append(("Flatten", f"[1, {flat_size}]"))
        
        code_lines.append(f"\n# Calculated Flatten Size: {current_c} * {current_h} * {current_w} = {flat_size}")
        code_lines.append("self.classifier = nn.Sequential(")
        code_lines.append("    nn.Flatten(),")
        code_lines.append(f"    nn.Linear({flat_size}, {num_classes})")
        code_lines.append(")")

elif model_type == "Multi-Layer Perceptron (MLP/SLP)":
    st.subheader("MLP Builder")
    col_layers, col_viz = st.columns([1, 1])
    
    with col_layers:
        st.info(f"Input is flattened automatically: {in_h}x{in_w}x{in_c} = {in_h*in_w*in_c} features.")
        
        num_hidden = st.slider("Number of Hidden Layers", 1, 5, 2)
        input_dim = in_h * in_w * in_c
        
        code_lines.append("self.net = nn.Sequential(")
        code_lines.append("    nn.Flatten(),")
        
        dims_log.append(("Input (Image)", f"[1, {in_c}, {in_h}, {in_w}]"))
        dims_log.append(("Flatten", f"[1, {input_dim}]"))
        
        for i in range(num_hidden):
            hidden_dim = st.select_slider(f"Layer {i+1} Neurons", options=[16, 32, 64, 128, 256, 512, 1024], value=128, key=f"h{i}")
            
            code_lines.append(f"    nn.Linear({input_dim}, {hidden_dim}),")
            code_lines.append("    nn.ReLU(),")
            
            dims_log.append((f"Linear_{i+1}", f"[1, {hidden_dim}]"))
            input_dim = hidden_dim # Set next input to current output
            
        # Final Output Layer
        code_lines.append(f"    nn.Linear({input_dim}, {num_classes})")
        dims_log.append((f"Output", f"[1, {num_classes}]"))
        code_lines.append(")")

# --- 3. VISUALIZATION & CODE EXPORT ---
with col_viz:
    st.subheader("Shape Debugger")
    
    # Check for invalid shapes (negative dimensions)
    valid_arch = True
    if model_type == "Convolutional (CNN)" and (current_h <= 0 or current_w <= 0):
        st.error(f"Error: Your kernel/stride reduces the image dimension to {current_h}x{current_w} (Negative or Zero). Reduce kernel size or remove pooling.")
        valid_arch = False
    
    # Display Dimensions Table
    df_log = pd.DataFrame(dims_log, columns=["Layer", "Output Tensor Shape"])
    st.table(df_log)
    
    if valid_arch:
        st.success("Architecture is Valid! ✅")
    
    st.subheader("Generated PyTorch Code")
    
    full_code = f"""
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        
{chr(10).join(['        ' + line for line in code_lines])}

    def forward(self, x):
        {'x = self.features(x)' if model_type == 'Convolutional (CNN)' else 'x = self.net(x)'}
        {'x = self.classifier(x)' if model_type == 'Convolutional (CNN)' else ''}
        return x

# Instantiate
model = CustomModel()
print(model)
"""
    st.code(full_code, language="python")

st.divider()
ask_ai()
