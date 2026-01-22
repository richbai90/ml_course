import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Function Approximation: Finding the Hidden Rules")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
dimension = st.sidebar.radio("Select Transformation Space", ["2D Transformation (Curve)", "3D Transformation (Surface)"])

lightbulb_callout("Big Picture", r'''Machine learning is fundamentally a function approximation problem. Given a dataset, we attempt to model an unknown underlying function $f: \mathbb{R}^n \to \mathbb{R}^m$. This module explores the trade-offs in
                  polynomial regression, specifically illustrating how increasing model complexity (degrees of freedom) reduces bias but increases variance—the core mechanism behind Overfitting and Underfitting.''') 

# ==========================================
# 2D TRANSFORMATION
# ==========================================
if dimension == "2D Transformation (Curve)":
    st.subheader("2D: Mapping Input $I$ to Output $O$")
    
    # 1. Generate Data (The "Unknown" Transformation + Noise)
    n_samples = st.sidebar.slider("Number of Samples", 10, 50, 20)
    noise_level = st.sidebar.slider("Noise (Corruption)", 0.0, 2.0, 0.5)
    
    # The Hidden Ground Truth Function (The Target T)
    x = np.linspace(-5, 5, n_samples)
    true_transformation = 0.5 * x + np.sin(x) * 2
    
    # Observed Data (Signal + Noise)
    np.random.seed(42) 
    noise = np.random.normal(0, noise_level, n_samples)
    y_observed = true_transformation + noise

    # 2. Approximate the Transformation (The Learner)
    degree = st.sidebar.slider("Model Complexity (Polynomial Degree)", 1, 15, 3)
    
    # Fit the polynomial
    coefficients = np.polyfit(x, y_observed, degree)
    model_poly = np.poly1d(coefficients)
    
    # Calculate the Model's "Transformed" points for the specific inputs
    y_model_points = model_poly(x)
    
    # Calculate smooth curve for visualization
    x_smooth = np.linspace(-5, 5, 200)
    y_model_smooth = model_poly(x_smooth)
    y_true_smooth = 0.5 * x_smooth + np.sin(x_smooth) * 2

    # 3. Visualization
    fig = go.Figure()

    # A. The Observed Data (Reality)
    fig.add_trace(go.Scatter(
        x=x, y=y_observed, 
        mode='markers', 
        name='Observed Data (Set O)',
        marker=dict(color='red', size=10, opacity=0.6, symbol='circle')
    ))

    # B. The Hidden Truth (Goal)
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_true_smooth, 
        mode='lines', 
        name='Hidden Transfer Function (T)',
        line=dict(color='green', dash='dash', width=2),
        opacity=0.5
    ))

    # C. The Model's Transformation (Approximation)
    fig.add_trace(go.Scatter(
        x=x, y=y_model_points, 
        mode='markers', 
        name='Model Transformed Points',
        marker=dict(color='blue', size=8, symbol='x')
    ))
    
    # D. The Model's Continuous Map
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_model_smooth, 
        mode='lines', 
        name=f'Learned Function (Degree {degree})',
        line=dict(color='blue', width=3, shape='spline'),
        opacity=0.8
    ))

    # E. Visualizing the Error/Transformation
    # Draw lines from Observed to Transformed to show what the model "thinks"
    x_lines = []
    y_lines = []
    for xi, y_obs, y_mod in zip(x, y_observed, y_model_points):
        x_lines.extend([xi, xi, None])
        y_lines.extend([y_obs, y_mod, None])

    fig.add_trace(go.Scatter(
        x=x_lines, y=y_lines,
        mode='lines',
        name='Transformation / Error',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=True,
        opacity=0.5
    ))

    fig.update_layout(
        title="Visualizing the Transformation",
        xaxis_title="Input Set {I}",
        yaxis_title="Output Set {O}",
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Educational Content (Updated)
    st.info(f"**Current Hypothesis: Degree {degree} Polynomial**")
    
    st.markdown("""
    Notice that increasing the complexity of the model improves the fit, but only to a point. If the model becomes too complex the error begins to grow again. This is what is known as **overfitting**. 
    
    Conversely, if the model is too simple, it cannot approximate the transformation effectively at all. This is what is known as **underfitting**. When designing and training a model, you must be aware of both potential issues.
    """)

# ==========================================
# 3D TRANSFORMATION
# ==========================================
else:
    st.subheader("3D: Mapping $(x, y)$ to Output $z$")
    
    # 1. Generate Data
    grid_size = st.sidebar.slider("Grid Resolution", 5, 15, 8)
    noise_3d = st.sidebar.slider("Noise Level", 0.0, 5.0, 2.0)
    
    # Create domain
    x_range = np.linspace(-5, 5, grid_size)
    y_range = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # The True Transformation (Hidden)
    # z = 0.5x^2 + 0.5y^2 + 5
    Z_true = 0.5 * X**2 + 0.5 * Y**2 + 5
    
    # Add noise to create the Observed Set
    np.random.seed(42)
    Z_observed = Z_true + np.random.normal(0, noise_3d, Z_true.shape)
    
    # Flatten for optimization
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z_observed.ravel()

    # 2. Define Candidate Transformation Family
    def transformation_function(data, a, b, c):
        x, y = data
        return a * x**2 + b * y**2 + c

    # Search for parameters a, b, c
    popt, _ = curve_fit(transformation_function, (x_flat, y_flat), z_flat)
    a_fit, b_fit, c_fit = popt
    
    # Calculate the estimated surface
    Z_fit = transformation_function((X, Y), a_fit, b_fit, c_fit)

    # 3. Visualization
    fig = go.Figure()

    # Observed Data
    fig.add_trace(go.Scatter3d(
        x=x_flat, y=y_flat, z=z_flat,
        mode='markers',
        name='Observed Data',
        marker=dict(size=5, color='red', opacity=0.6)
    ))

    # Transformed Points (Where the model maps these inputs)
    fig.add_trace(go.Scatter3d(
        x=x_flat, y=y_flat, z=Z_fit.ravel(),
        mode='markers',
        name='Transformed Points',
        marker=dict(size=4, color='blue', symbol='x')
    ))

    # Learned Transformation Surface
    fig.add_trace(go.Surface(
        z=Z_fit, x=X, y=Y,
        name='Learned Surface',
        colorscale='Viridis',
        opacity=0.6,
        showscale=False
    ))

    # Connect Observed to Transformed (Residuals)
    x_lines = []
    y_lines = []
    z_lines = []
    for i in range(len(x_flat)):
        x_lines.extend([x_flat[i], x_flat[i], None])
        y_lines.extend([y_flat[i], y_flat[i], None])
        z_lines.extend([z_flat[i], Z_fit.ravel()[i], None])

    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        name='Error',
        line=dict(color='gray', width=2),
        showlegend=True
    ))

    fig.update_layout(
        title=f"Learned Map: z ≈ {a_fit:.2f}x² + {b_fit:.2f}y² + {c_fit:.2f}",
        scene=dict(
            xaxis_title='Input X',
            yaxis_title='Input Y',
            zaxis_title='Output Z'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# EXPLANATION
# ==========================================
st.divider()

st.markdown(r"""
### What is a Neural Network?
            
A neural network is a function approximator. Given 2 sets of data $\left\{I\right\}$ and $\left\{O\right\}$, we assume
that there exists a function $T(X \in I)$ that satisfies the condition
$$T(X \in I) \longrightarrow Y \in \left\{O\right\}$$
            
If our assumption is accurate, and such a mapping does exist, the job of the neural network is to approximate it.

### What is machine learning?

Machine learning is the method used to find the the unknown function T.
There are several methods, but the most popular by far is gradient descent optimization.
""")

st.divider()
ask_ai()
