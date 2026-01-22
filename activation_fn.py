import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

from utils import lightbulb_callout

st.set_page_config(layout="wide")
st.title("Theory: Activation Functions")
lightbulb_callout("Big Picture", r'''Without non-linear activation functions, a deep neural network collapses mathematically into a single linear transformation (since the product of matrices is just another matrix). This module examines functions like
                  ReLU and Sigmoid, which introduce the necessary non-linearities to approximate complex, non-convex functions.''')

st.markdown("""
**The "Spark" of Intelligence.** Without activation functions, a Neural Network is just a giant Linear Regression model. No matter how many layers you stack, $W_2(W_1(x))$ is mathematically just a single matrix $W_{new}$.

Activation functions introduce **Non-Linearity**, allowing the network to learn curves, decision boundaries, and complex structures.
""")

# --- TABS ---
tab1, tab2 = st.tabs(["1. The Zoo (Formulas & Gradients)", "2. Visualizing Non-Linearity"])

# ==========================================
# TAB 1: THE ZOO
# ==========================================
with tab1:
    col_sel, col_plot = st.columns([1, 2])
    
    with col_sel:
        st.subheader("Select a Function")
        act_choice = st.radio(
            "Activation", 
            ["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "GELU", "Swish"],
            index=2
        )
        
        desc = {
            "Sigmoid": "**Classic.** Squishes numbers between 0 and 1. Used in binary classification output. **Problem:** Vanishing Gradient (derivative kills learning at extremes).",
            "Tanh": "**Zero-Centered.** Squishes numbers between -1 and 1. Generally better than Sigmoid for hidden layers.",
            "ReLU": "**The King.** $max(0, x)$. Fast to compute, no vanishing gradient for positive values. **Problem:** Dead Relu (if negative, gradient is 0).",
            "LeakyReLU": "**The Fix.** Allows a tiny gradient for negative values so neurons don't 'die'.",
            "GELU": "**Modern Standard.** Used in Transformers (BERT, GPT). A smooth approximation of ReLU.",
            "Swish": "**Google's Invention.** $x \\cdot \\sigma(x)$. Self-gated activation function."
        }
        
        st.info(desc[act_choice])

    with col_plot:
        # Generate Data
        x = torch.linspace(-6, 6, 200)
        x.requires_grad = True
        
        # Apply Function
        if act_choice == "Sigmoid": y = torch.sigmoid(x)
        elif act_choice == "Tanh": y = torch.tanh(x)
        elif act_choice == "ReLU": y = F.relu(x)
        elif act_choice == "LeakyReLU": y = F.leaky_relu(x, negative_slope=0.1)
        elif act_choice == "GELU": y = F.gelu(x)
        elif act_choice == "Swish": y = F.silu(x) # SiLU is Swish with beta=1
        
        # Calculate Derivative (Gradient)
        y.sum().backward()
        grad = x.grad
        
        # Convert to Numpy
        x_np = x.detach().numpy()
        y_np = y.detach().numpy()
        grad_np = grad.detach().numpy()
        
        # Plot
        fig = go.Figure()
        
        # 1. The Function
        fig.add_trace(go.Scatter(x=x_np, y=y_np, name=f"f(x) = {act_choice}", line=dict(color='blue', width=4)))
        
        # 2. The Derivative
        fig.add_trace(go.Scatter(x=x_np, y=grad_np, name="f'(x) Gradient", line=dict(color='red', dash='dot', width=2)))
        
        fig.update_layout(
            title=f"{act_choice} and its Derivative",
            xaxis_title="Input (x)",
            yaxis_title="Output (y)",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: SPACE FOLDING DEMO
# ==========================================
with tab2:
    st.subheader("How Activations Warps Space")
    st.markdown("A neural network layer performs two steps: **Rotate/Skew** (Linear Matrix) $\\to$ **Fold/Squish** (Activation).")
    
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        demo_act = st.selectbox("Activation to Apply", ["ReLU", "Tanh", "Sigmoid", "None (Linear)"])
        
        st.write("---")
        st.caption("Apply a random linear transformation matrix $W$ before activation:")
        if st.button("Randomize Matrix W"):
            st.session_state['W'] = np.random.randn(2, 2)
            
        if 'W' not in st.session_state:
            st.session_state['W'] = np.eye(2) # Identity by default
            
        st.write(f"Current Matrix W:\n{np.round(st.session_state['W'], 2)}")

    with col_viz:
        # Create a grid of points
        grid_size = 20
        range_val = 4
        x = np.linspace(-range_val, range_val, grid_size)
        y = np.linspace(-range_val, range_val, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Flatten to (N, 2)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        
        # 1. Linear Transform (Matrix Mult)
        transformed = points @ st.session_state['W']
        
        # 2. Activation
        if demo_act == "ReLU":
            transformed = np.maximum(0, transformed)
        elif demo_act == "Tanh":
            transformed = np.tanh(transformed)
        elif demo_act == "Sigmoid":
            transformed = 1 / (1 + np.exp(-transformed))
            
        # Reshape for plotting lines (Meshgrid style)
        X_new = transformed[:, 0].reshape(grid_size, grid_size)
        Y_new = transformed[:, 1].reshape(grid_size, grid_size)
        
        fig_warp = go.Figure()
        
        # Plot "Horizontal" lines of the grid
        for i in range(grid_size):
            fig_warp.add_trace(go.Scatter(x=X_new[i, :], y=Y_new[i, :], mode='lines', line=dict(color='gray', width=1), showlegend=False))
            
        # Plot "Vertical" lines of the grid
        for i in range(grid_size):
            fig_warp.add_trace(go.Scatter(x=X_new[:, i], y=Y_new[:, i], mode='lines', line=dict(color='gray', width=1), showlegend=False))

        # Add origin point for reference
        fig_warp.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='red', size=10), name="Origin"))

        fig_warp.update_layout(
            title=f"Grid after Wx + {demo_act}",
            xaxis=dict(range=[-5, 5], title="x1 output"),
            yaxis=dict(range=[-5, 5], title="x2 output"),
            height=500,
            width=500
        )
        st.plotly_chart(fig_warp)
        
        if demo_act == "ReLU":
            st.caption("Notice how ReLU 'folds' all negative space onto the axes? That's how it creates piecewise linear boundaries.")
        elif demo_act == "Tanh":
            st.caption("Notice how Tanh 'squishes' the infinite grid into a 1x1 box.")
