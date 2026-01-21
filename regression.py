import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(layout="wide")
st.title("Lab: Regression & Model Capacity")

st.markdown("""
**Can your network learn this shape?**
This lab focuses on **Model Capacity**. 
* If the network is too simple (e.g., linear activation), it can't learn curves.
* If the network is too shallow, it might struggle with complex frequencies.
""")

# --- 1. GLOBAL CONFIGURATION (Sidebar) ---
# We keep inputs outside the fragment so they are accessible globally
with st.sidebar:
    st.header("1. The Problem (Data)")
    func_type = st.selectbox("Target Function", ["Line (y=mx+b)", "Parabola (y=x^2)", "Sine Wave", "Step Function", "Absolute Value"])
    noise_scale = st.slider("Noise Level", 0.0, 0.3, 0.05)
    n_samples = st.slider("Number of Points", 50, 500, 200)
    
    st.header("2. The Architect (Model)")
    n_layers = st.slider("Hidden Layers", 1, 5, 2)
    n_neurons = st.slider("Neurons per Layer", 4, 128, 32)
    activation_name = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
    
    st.header("3. Training Config")
    lr = st.selectbox("Learning Rate", [0.1, 0.01, 0.005, 0.001], index=1)
    epochs = st.slider("Epochs", 100, 2000, 500)

# --- 2. DATA & MODEL DEFINITIONS ---
@st.cache_data
def generate_regression_data(func_type, n, noise):
    x = np.linspace(-3, 3, n)
    
    if func_type == "Line (y=mx+b)":
        y = 0.5 * x + 0.2
    elif func_type == "Parabola (y=x^2)":
        y = x**2 - 1
    elif func_type == "Sine Wave":
        y = np.sin(2 * x)
    elif func_type == "Step Function":
        y = np.where(x > 0, 1.0, 0.0)
    elif func_type == "Absolute Value":
        y = np.abs(x)
        
    # Add noise
    y_noisy = y + np.random.normal(0, noise, n)
    return x, y, y_noisy

class DynamicMLP(nn.Module):
    def __init__(self, layers, neurons, act_name):
        super().__init__()
        if act_name == "ReLU": self.act = nn.ReLU()
        elif act_name == "Tanh": self.act = nn.Tanh()
        elif act_name == "Sigmoid": self.act = nn.Sigmoid()
        elif act_name == "LeakyReLU": self.act = nn.LeakyReLU()
        
        layer_list = []
        layer_list.append(nn.Linear(1, neurons))
        layer_list.append(self.act)
        for _ in range(layers - 1):
            layer_list.append(nn.Linear(neurons, neurons))
            layer_list.append(self.act)
        layer_list.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.net(x)

# --- 3. THE LAB FRAGMENT ---
# This decorator isolates the function. When the button inside triggers a rerun,
# ONLY this function reruns, not the whole page!
@st.fragment
def run_lab_interface():
    # Fetch Data (Cached)
    x_np, y_true_np, y_noisy_np = generate_regression_data(func_type, n_samples, noise_scale)
    X = torch.FloatTensor(x_np).unsqueeze(1)
    Y = torch.FloatTensor(y_noisy_np).unsqueeze(1)

    col_viz, col_stats = st.columns([3, 1])

    # -- Stats & Controls --
    with col_stats:
        st.subheader("Controls")
        start_btn = st.button("Start Training", type="primary")
        
        st.subheader("Live Metrics")
        epoch_placeholder = st.empty()
        loss_placeholder = st.empty()
        
        if activation_name in ["Sigmoid", "Tanh"] and func_type in ["Step Function", "Absolute Value"]:
            st.warning(f"⚠️ {activation_name} struggles with sharp corners!")

    # -- Visualization Placeholder --
    with col_viz:
        chart_placeholder = st.empty()

    # -- Training Loop --
    if start_btn:
        model = DynamicMLP(n_layers, n_neurons, activation_name)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Smart Step Size (Aim for ~30 updates total for smoothness)
        step_size = max(1, epochs // 30)
        
        for epoch in range(epochs + 1):
            # Optimization
            model.train()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, Y)
            loss.backward()
            optimizer.step()
            
            # UI Update
            if epoch % step_size == 0 or epoch == epochs:
                loss_val = loss.item()
                
                # Update Metrics
                epoch_placeholder.metric("Epoch", f"{epoch}/{epochs}")
                loss_placeholder.metric("Loss", f"{loss_val:.5f}")
                
                # Update Chart
                with torch.no_grad():
                    pred_np = y_pred.numpy().flatten()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_np, y=y_noisy_np, mode='markers', name='Data', marker=dict(color='gray', opacity=0.4)))
                fig.add_trace(go.Scatter(x=x_np, y=y_true_np, mode='lines', name='Truth', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=x_np, y=pred_np, mode='lines', name='Model', line=dict(color='red', width=3)))
                
                fig.update_layout(
                    title=f"Fitting {func_type} with {activation_name}",
                    xaxis_title="Input (x)", yaxis_title="Output (y)",
                    height=500,
                    yaxis=dict(range=[-2, 3]) # Fixed range reduces jitter
                )
                
                # CRITICAL FIX: No 'key' argument here!
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Tiny sleep to allow the render thread to catch up
                time.sleep(0.01)

    else:
        # Static Initial View
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_np, y=y_noisy_np, mode='markers', name='Data', marker=dict(color='gray', opacity=0.4)))
        fig.add_trace(go.Scatter(x=x_np, y=y_true_np, mode='lines', name='Truth', line=dict(color='green', dash='dash')))
        fig.update_layout(title="Target Dataset", height=500)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# Run the fragment
run_lab_interface()
