import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Lab: Overfitting & Regularization")
lightbulb_callout('Big Picture', r'''Monitoring the divergence between Empirical Risk (Training Loss) and Expected Risk (Validation Loss). This lab simulates the training dynamics to identify the onset of overfitting, introducing Regularization techniques
                  (like L2 weight decay) to constrain the model's hypothesis space.''')
st.markdown("""
**The Battle of Bias vs. Variance.**
* **Underfitting (High Bias):** The model is too simple to learn the pattern.
* **Overfitting (High Variance):** The model is so complex it memorizes the *noise* in the training data, failing to generalize to new data.

**Your Goal:** Train the model until the **Validation Loss** is low, but stop before it starts rising again (which indicates overfitting).
""")

# --- 1. CONFIGURATION SIDEBAR ---
with st.sidebar:
    st.header("1. Data Settings")
    n_samples = st.slider("Training Samples", 10, 200, 30, help="Fewer samples make overfitting easier.")
    noise_level = st.slider("Noise Level", 0.0, 0.5, 0.2)
    
    st.header("2. Model Architecture")
    hidden_size = st.slider("Hidden Neurons", 5, 200, 100, help="More neurons = Higher Capacity to memorize.")
    
    st.header("3. Training Config")
    learning_rate = st.selectbox("Learning Rate", [0.1, 0.01, 0.001], index=1)
    weight_decay = st.select_slider("L2 Regularization (Weight Decay)", 
                                    options=[0.0, 1e-4, 1e-3, 1e-2, 1e-1], 
                                    value=0.0,
                                    help="Penalizes large weights to prevent overfitting.")
    epochs = st.slider("Max Epochs", 100, 2000, 1000)

# --- 2. DATA GENERATION ---
@st.cache_data
def generate_data(n, noise):
    # True Function: Sin wave
    x = np.linspace(-3, 3, 200)
    y_true = np.sin(x * 2)
    
    # Training Data (Sparse & Noisy)
    np.random.seed(42)
    x_train = np.random.uniform(-3, 3, n)
    y_train = np.sin(x_train * 2) + np.random.normal(0, noise, n)
    
    # Validation Data (Clean/Dense check of generalization)
    x_val = np.random.uniform(-3, 3, 100)
    y_val = np.sin(x_val * 2) + np.random.normal(0, noise, 100)
    
    return x, y_true, x_train, y_train, x_val, y_val

x_plot, y_plot, x_train, y_train, x_val, y_val = generate_data(n_samples, noise_level)

# Convert to PyTorch
X_train_t = torch.FloatTensor(x_train).unsqueeze(1)
Y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(x_val).unsqueeze(1)
Y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
X_plot_t = torch.FloatTensor(x_plot).unsqueeze(1)

# --- 3. MODEL DEFINITION ---
class SimpleMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- 4. MAIN INTERFACE ---
col_charts, col_metrics = st.columns([3, 1])

start_btn = st.sidebar.button("Start Training", type="primary")
stop_btn = st.sidebar.button("Stop")

if start_btn:
    # Initialize
    model = SimpleMLP(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    # Placeholders for live updates
    chart_loss_placeholder = col_charts.empty()
    chart_fit_placeholder = col_charts.empty()
    metric_placeholder = col_metrics.empty()
    
    # Training Loop
    for epoch in range(epochs):
        # -- Train Step --
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, Y_train_t)
        loss.backward()
        optimizer.step()
        
        # -- Validation Step --
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, Y_val_t)
            
            # Prediction for visualization (Dense grid)
            y_plot_pred = model(X_plot_t).numpy().flatten()
            
        # Record
        t_loss = loss.item()
        v_loss = val_loss.item()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        # -- Live Update (every 10 epochs to save rendering time) --
        if epoch % 10 == 0 or epoch == epochs - 1:
            
            # 1. Loss Curve
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss', line=dict(color='blue')))
            fig_loss.add_trace(go.Scatter(y=val_losses, mode='lines', name='Val Loss', line=dict(color='red')))
            fig_loss.update_layout(title="Loss Curves (Lower is Better)", xaxis_title="Epoch", yaxis_title="MSE Loss", height=300)
            chart_loss_placeholder.plotly_chart(fig_loss, use_container_width=True)
            
            # 2. Function Fit
            fig_fit = go.Figure()
            # Ground Truth
            fig_fit.add_trace(go.Scatter(x=x_plot, y=y_plot, name='True Function', line=dict(color='green', dash='dash', width=1)))
            # Training Data
            fig_fit.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Training Data', marker=dict(color='blue', size=8)))
            # Model Prediction
            fig_fit.add_trace(go.Scatter(x=x_plot, y=y_plot_pred, name='Model Prediction', line=dict(color='red', width=3)))
            
            fig_fit.update_layout(title=f"Model Fit at Epoch {epoch}", xaxis_title="Input (x)", yaxis_title="Output (y)", height=400)
            chart_fit_placeholder.plotly_chart(fig_fit, use_container_width=True)
            
            # 3. Metrics
            metric_placeholder.markdown(f"""
            ### Epoch {epoch}/{epochs}
            * **Train Loss:** {t_loss:.4f}
            * **Val Loss:** {v_loss:.4f}
            
            ---
            **Status:** {'Overfitting!' if v_loss > t_loss * 1.5 and epoch > 50 else 'Training...'}
            """)
            
            time.sleep(0.01) # Small delay for UI smoothness
            
else:
    st.info("Adjust parameters on the left and click 'Start Training' to visualize the dynamics.")
    
    # Show static initial state
    fig_static = go.Figure()
    fig_static.add_trace(go.Scatter(x=x_plot, y=y_plot, name='True Function', line=dict(color='green', dash='dash')))
    fig_static.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Training Data', marker=dict(color='blue')))
    fig_static.update_layout(title="Initial Data Distribution", height=400)
    col_charts.plotly_chart(fig_static, use_container_width=True)

st.divider()
ask_ai()
