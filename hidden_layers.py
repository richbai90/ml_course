import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(layout='wide')
st.title("The Hidden Layer: Bending Space")
st.markdown(r"""
We've seen that a single layer (like linear regression) draws a straight line (or plane) to separate data. 
But what if the data **cannot** be separated by a straight line?

This is where the **Hidden Layer** comes in. It acts as an intermediate transformation that "warps" or "lifts" the input data into a new space where it becomes easy to separate.
""")

# --- Sidebar ---
st.sidebar.header("Network Config")
num_hidden = 3 # Fixed for 3D visualization purposes
activation_type = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])

# --- 1. Generate Non-Linear Data (The "Bullseye") ---
n_samples = 100
np.random.seed(42)

# Generate concentric circles
# Inner circle (Class 0)
inner_r = np.random.uniform(0, 0.5, n_samples // 2)
inner_theta = np.random.uniform(0, 2*np.pi, n_samples // 2)
X_inner = np.column_stack([inner_r * np.cos(inner_theta), inner_r * np.sin(inner_theta)])
y_inner = np.zeros(n_samples // 2)

# Outer ring (Class 1)
outer_r = np.random.uniform(0.8, 1.3, n_samples // 2)
outer_theta = np.random.uniform(0, 2*np.pi, n_samples // 2)
X_outer = np.column_stack([outer_r * np.cos(outer_theta), outer_r * np.sin(outer_theta)])
y_outer = np.ones(n_samples // 2)

X = np.vstack([X_inner, X_outer])
y = np.hstack([y_inner, y_outer])

# --- 2. Define the Neural Network ---
# Structure: Input (2) -> Hidden (3) -> Output (1)
# We use 3 hidden neurons so we can plot the "Hidden State" in 3D.

def get_activation(x, type):
    if type == "ReLU":
        return np.maximum(0, x)
    elif type == "Tanh":
        return np.tanh(x)
    else: # Sigmoid
        return 1 / (1 + np.exp(-x))

def forward_pass(params, X_in, return_hidden=False):
    # Unpack weights
    # W1: 2 inputs -> 3 hidden (size 6)
    # b1: 3 hidden biases (size 3)
    # W2: 3 hidden -> 1 output (size 3)
    # b2: 1 output bias (size 1)
    W1 = params[:6].reshape(2, 3)
    b1 = params[6:9].reshape(3)
    W2 = params[9:12].reshape(3, 1)
    b2 = params[12]
    
    # Hidden Layer Computations
    Z1 = np.dot(X_in, W1) + b1
    A1 = get_activation(Z1, activation_type) # The "Transformed" Data
    
    # Output Layer Computations
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2)) # Sigmoid output for classification
    
    if return_hidden:
        return A1, A2
    return A2

# Loss Function (Binary Cross Entropy)
def loss_function(params, X_in, y_true):
    y_pred = forward_pass(params, X_in).flatten()
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- 3. Train the Network (Optimization) ---
if "params" not in st.session_state:
    # Initialize random weights if not present
    st.session_state.params = np.random.randn(13) * 0.5

# Button to trigger training
train_clicked = st.sidebar.button("Train Network (100 iters)")

if train_clicked:
    # Use scipy to minimize loss (Gradient Descent equivalent)
    res = minimize(loss_function, st.session_state.params, args=(X, y), 
                   method='L-BFGS-B', options={'maxiter': 100})
    st.session_state.params = res.x

# Get predictions and hidden states with current weights
hidden_states, predictions = forward_pass(st.session_state.params, X, return_hidden=True)
predictions = predictions.flatten()
prediction_class = (predictions > 0.5).astype(int)

# --- 4. Visualization ---

col1, col2 = st.columns([1, 1])

# Plot A: Input Space (2D)
with col1:
    fig_input = go.Figure()
    
    # Scatter points colored by Class
    fig_input.add_trace(go.Scatter(
        x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Class 0 (Inner)',
        marker=dict(color='red', size=8)
    ))
    fig_input.add_trace(go.Scatter(
        x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Class 1 (Outer)',
        marker=dict(color='blue', size=8)
    ))

    # Optional: Draw Decision Boundary grid
    # (Computationally expensive to do dense grid, so we skip for speed or use low res)
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
    grid_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_preds = forward_pass(st.session_state.params, grid_flat).reshape(grid_x.shape)
    
    fig_input.add_trace(go.Contour(
        x=np.linspace(-1.5, 1.5, 50), y=np.linspace(-1.5, 1.5, 50), z=grid_preds,
        showscale=False, opacity=0.3, colorscale='RdBu',
        name='Decision Boundary'
    ))

    fig_input.update_layout(
        title="1. Input Space (2D)",
        xaxis_title="X1", yaxis_title="X2",
        width=400, height=400, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_input, use_container_width=True)

# Plot B: Hidden Space (3D)
with col2:
    fig_hidden = go.Figure()
    
    # We plot the 3 neuron activations as x, y, z coordinates
    # This visualizes "Where the data moved to"
    fig_hidden.add_trace(go.Scatter3d(
        x=hidden_states[y==0, 0], 
        y=hidden_states[y==0, 1], 
        z=hidden_states[y==0, 2],
        mode='markers', name='Class 0',
        marker=dict(color='red', size=5, opacity=0.8)
    ))
    fig_hidden.add_trace(go.Scatter3d(
        x=hidden_states[y==1, 0], 
        y=hidden_states[y==1, 1], 
        z=hidden_states[y==1, 2],
        mode='markers', name='Class 1',
        marker=dict(color='blue', size=5, opacity=0.8)
    ))
    
    fig_hidden.update_layout(
        title="2. Hidden Layer Space (3D)",
        scene=dict(
            xaxis_title="Neuron 1",
            yaxis_title="Neuron 2",
            zaxis_title="Neuron 3"
        ),
        width=400, height=400, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_hidden, use_container_width=True)

# --- Explanation ---
st.divider()
st.markdown(r"""
### What is happening?

1.  **Input Space (Left):** The data is a "bullseye". You cannot draw a single straight line to separate the Red dots from the Blue dots. A simple Linear Model would fail here (Underfitting).
2.  **Transformation:** The data passes through the **Hidden Layer**. The network multiplies the input by weights and applies the `{}` function.
3.  **Hidden Space (Right):** Look at the 3D plot. The neural network has learned to "lift" or "distort" the red center points away from the blue outer points. 
    * In this new 3D space, the two classes **are** linearly separable! A flat plane can now slice between them.
4.  **Output:** The final layer just draws that plane in the 3D space, which looks like a complex curve when projected back down to the 2D Input Space.

**Key Takeaway:** Hidden layers are "Feature Extractors." They transform input data that is hard to solve into a new representation where the answer is easy.
""".format(activation_type))

# Metric
accuracy = np.mean(prediction_class == y)
st.metric("Model Accuracy", f"{accuracy:.0%}")
