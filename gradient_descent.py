import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")
st.title("Gradient Descent: Walking Down the Mountain")
st.markdown(r"""
**The Goal:** We defined our Neural Network's performance using a **Loss Function** (Error). The lower the loss, the better the model.

**The Problem:** We don't know the perfect weights. We are standing somewhere on a "Loss Landscape" and need to find the lowest point (Global Minimum).

**The Algorithm:**
$$w_{new} = w_{old} - \alpha \cdot \nabla J(w)$$

* $\nabla J(w)$: The **Gradient** (Steepness). It points *uphill*.
* $-$ (Minus): We want to go *downhill*.
* $\alpha$ (Alpha): The **Learning Rate**. How big is the step we take?
""")

# --- Sidebar Controls ---
st.sidebar.header("Hyperparameters")

# 1. Choose Landscape
landscape_type = st.sidebar.selectbox(
    "Loss Landscape", 
    ["Convex Bowl (Easy)", "Non-Convex Wavy (Hard)"]
)

# 2. Algorithm Settings
lr = st.sidebar.slider("Learning Rate (Alpha)", 0.01, 1.2, 0.1, step=0.01)
iterations = st.sidebar.slider("Iterations (Steps)", 5, 50, 20)

# 3. Starting Position
start_x = st.sidebar.slider("Start X", -4.0, 4.0, 3.5)
start_y = st.sidebar.slider("Start Y", -4.0, 4.0, 3.5)

# --- 1. Define Functions & Gradients ---

def calculate_loss_and_gradient(x, y, type):
    """
    Returns (Loss, Gradient_X, Gradient_Y)
    """
    if type == "Convex Bowl (Easy)":
        # f(x,y) = x^2 + y^2
        loss = x**2 + y**2
        grad_x = 2*x
        grad_y = 2*y
        return loss, grad_x, grad_y
        
    else: # Non-Convex Wavy
        # f(x,y) = x^2/4 + y^2/4 - cos(2x) - cos(2y) + 2
        # A bowl with bumps (local minima)
        loss = (x**2)/4 + (y**2)/4 - np.cos(2*x) - np.cos(2*y) + 2
        
        # Derivatives:
        # df/dx = x/2 + 2sin(2x)
        grad_x = (x/2) + 2*np.sin(2*x)
        grad_y = (y/2) + 2*np.sin(2*y)
        return loss, grad_x, grad_y

# --- 2. Run Gradient Descent ---
path_x = [start_x]
path_y = [start_y]
path_z = []

# Initial Loss
l, _, _ = calculate_loss_and_gradient(start_x, start_y, landscape_type)
path_z.append(l)

curr_x, curr_y = start_x, start_y

for i in range(iterations):
    _, gx, gy = calculate_loss_and_gradient(curr_x, curr_y, landscape_type)
    
    # The Update Rule
    curr_x = curr_x - lr * gx
    curr_y = curr_y - lr * gy
    
    # Calculate new z for plotting
    loss, _, _ = calculate_loss_and_gradient(curr_x, curr_y, landscape_type)
    
    path_x.append(curr_x)
    path_y.append(curr_y)
    path_z.append(loss)

# --- 3. Visualization (3D Surface) ---

# Generate Grid for Surface Plot
grid_range = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(grid_range, grid_range)

if landscape_type == "Convex Bowl (Easy)":
    Z = X**2 + Y**2
else:
    Z = (X**2)/4 + (Y**2)/4 - np.cos(2*X) - np.cos(2*Y) + 2

fig = go.Figure()

# A. The Landscape
fig.add_trace(go.Surface(
    z=Z, x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='Loss Surface',
    showscale=False,
))

# B. The Path (The Ball Rolling Down)
fig.add_trace(go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode='lines+markers',
    name='Gradient Descent Path',
    line=dict(color='red', width=5),
    marker=dict(size=4, color='yellow')
))

# C. The Start Point
fig.add_trace(go.Scatter3d(
    x=[path_x[0]], y=[path_y[0]], z=[path_z[0]],
    mode='markers',
    name='Start',
    marker=dict(size=8, color='green', symbol='diamond')
))

# D. The End Point
fig.add_trace(go.Scatter3d(
    x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
    mode='markers',
    name='End',
    marker=dict(size=8, color='red', symbol='x')
))

fig.update_layout(
    title="Visualizing Optimization",
    scene=dict(
        xaxis_title='Weight 1',
        yaxis_title='Weight 2',
        zaxis_title='Loss (Error)'
    ),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# --- Educational Feedback ---
final_loss = path_z[-1]
st.info(f"**Final Status:** Reached Loss = {final_loss:.4f} after {iterations} steps.")

if landscape_type == "Non-Convex Wavy (Hard)":
    st.markdown("""
    ### Try This:
    1.  Set **Learning Rate** to `0.01` (Small). Notice how slow it is? It might run out of iterations before reaching the bottom.
    2.  Set **Learning Rate** to `1.0` (Large). Notice how it jumps wildly? It might overshoot the hole entirely.
    3.  Move **Start X / Y** around. Can you get it stuck in a "Local Minimum" (a small valley that isn't the true bottom)?
    """)
else:
    st.markdown("""
    ### Try This:
    1.  Increase the **Learning Rate** above `1.0`. Watch the path oscillate back and forth.
    2.  If the Learning Rate is too high, the algorithm will **diverge** (explode outwards) instead of converging.
    """)
