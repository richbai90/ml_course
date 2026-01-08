import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Interactive 1D Convolution Lesson")

# --- Parameters ---
t = np.linspace(0, 10, 500)
dt = t[1] - t[0]
sine_wave = np.sin(t)

# Define a simple boxcar window (h(t))
window_width = 50
h = np.zeros_like(t)
h[:window_width] = 1.0
# Normalize window for area = 1
h = h / (window_width * dt)

# Pre-calculate convolution
# 'same' ensures the output length matches the input length
conv_result = np.convolve(sine_wave, h, mode="same") * dt

# --- Sidebar / Controls ---
st.sidebar.header("Controls")
step = st.sidebar.slider("Slide the window (t)", 0, len(t) - 1, 0)

# --- Visualization ---
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(
        "Input Signal: x(t) = sin(t)",
        "Moving Window: h(t - τ)",
        "Convolution Result: (x * h)(t)",
    ),
)

# 1. Sine Wave
fig.add_trace(
    go.Scatter(x=t, y=sine_wave, name="Sine Wave", line=dict(color="blue")),
    row=1,
    col=1,
)

# 2. Moving Window
# We shift the window by centering it on the current 'step'
shifted_h = np.zeros_like(t)
half_w = window_width // 2
start = max(0, step - half_w)
end = min(len(t), step + half_w)
shifted_h[start:end] = h[0 : (end - start)]

fig.add_trace(
    go.Scatter(
        x=t, y=shifted_h, name="Window", fill="tozeroy", line=dict(color="orange")
    ),
    row=2,
    col=1,
)

# 3. Cumulative Convolution
# We only show the result up to the current 'step'
fig.add_trace(
    go.Scatter(
        x=t[:step],
        y=conv_result[:step],
        name="Result",
        line=dict(color="green", width=3),
    ),
    row=3,
    col=1,
)

# Keep the X and Y axes consistent
fig.update_xaxes(range=[0, 10])
fig.update_yaxes(range=[-1.5, 1.5], row=1, col=1)
fig.update_yaxes(range=[-0.5, 2.5], row=2, col=1)
fig.update_yaxes(range=[-1.5, 1.5], row=3, col=1)

fig.update_layout(height=800, showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# --- Explanation ---
st.markdown(
    f"""
### How it works:
1. **The Window:** As you move the slider, the window $h(t)$ slides across the sine wave.
2. **The Calculation:** At each point, the convolution is the integral of the product of these two functions.
3. **The Result:** Notice how the convolution "smooths" the sine wave or shifts its phase depending on the window width. 
   Currently, the window is at index **{step}**.
"""
)
