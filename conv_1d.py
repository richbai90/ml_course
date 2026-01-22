import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal

from utils import lightbulb_callout

st.set_page_config(layout="wide")
st.title("1D Convolution: The Full Interaction")

lightbulb_callout("Big Picture", r'''A formal look at the convolution integral $(f * g)(t)$. You will visualize the standard "flip and shift" operation used in signal processing to demonstrate how learnable kernels act as matched filters, maximizing their
                  response when the signal aligns with the kernel's features.''')

# --- 1. Setup Time and Signals ---
n_pts = 500
t = np.linspace(0, 10, n_pts)
dt = t[1] - t[0]
x_t = np.sin(t)

# --- 2. Sidebar Controls ---
st.sidebar.header("Kernel Settings")
kernel_type = st.sidebar.selectbox(
    "Select Function h(t)", ["Triangle (Window)", "Sawtooth (Window)", "Full Sine Wave"]
)

# --- 3. Construct h(t) ---
h_t = np.zeros_like(t)

if kernel_type == "Triangle (Window)":
    width = 100
    h_raw = signal.windows.triang(width)
    h_t[:width] = h_raw / (np.sum(h_raw) * dt)
elif kernel_type == "Sawtooth (Window)":
    width = 100
    h_raw = np.linspace(0, 1, width)
    h_t[:width] = h_raw / (np.sum(h_raw) * dt)
else:  # Full Sine Wave
    phase_shift = np.pi / 4
    h_t = np.sin(t + phase_shift)

# --- 4. The Convolution Math ---
conv_result = np.convolve(x_t, h_t, mode="full") * dt
t_conv = np.linspace(0, 20, len(conv_result))

# --- 5. Interactive Slider ---
# Now 'step' spans the entire length of the convolution result
step = st.slider("Slide h(t) across x(t)", 0, len(t_conv) - 1, 0)

# --- 6. The "World Space" Visualization ---
# We create a 'canvas' of length t_conv (approx 1000 points)
# x_t is stationary in the middle (from t=5 to t=15)
x_canvas = np.zeros(len(t_conv))
x_canvas[n_pts - 1 : 2 * n_pts - 1] = x_t

# h_flipped starts at the left and slides right
h_flipped = h_t[::-1]
h_sliding = np.zeros(len(t_conv))

# Calculate slice for sliding kernel
start_idx = step
end_idx = step + n_pts
if end_idx > len(t_conv):
    end_idx = len(t_conv)
    h_slice = h_flipped[: (end_idx - start_idx)]
else:
    h_slice = h_flipped

h_sliding[start_idx:end_idx] = h_slice

# --- 7. Plotting ---
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Stationary Function: x(τ)",
        f"Flipped & Sliding: h(t - τ)",
        "Result: (x * h)(t)",
    ),
)

# Plot 1: Stationary Signal x(τ)
fig.add_trace(
    go.Scatter(x=t_conv, y=x_canvas, name="x(τ)", line=dict(color="#636EFA")),
    row=1,
    col=1,
)

# Plot 2: Sliding Function h(t-τ)
fig.add_trace(
    go.Scatter(
        x=t_conv, y=h_sliding, name="h(t-τ)", fill="tozeroy", line=dict(color="#EF553B")
    ),
    row=2,
    col=1,
)

# Plot 3: Result
# We plot the full line in light grey and the "active" part in green
fig.add_trace(
    go.Scatter(
        x=t_conv,
        y=conv_result,
        name="Full Path",
        line=dict(color="rgba(200, 200, 200, 0.3)", dash="dot"),
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=t_conv[:step],
        y=conv_result[:step],
        name="Current Result",
        line=dict(color="#00CC96", width=4),
    ),
    row=3,
    col=1,
)

# Formatting
fig.update_xaxes(range=[0, 20], title_text="Time (t)")
fig.update_yaxes(range=[-1.5, 1.5], row=1, col=1)
fig.update_yaxes(range=[-1.5, 1.5], row=2, col=1)
res_max = np.max(np.abs(conv_result))
fig.update_yaxes(range=[-res_max * 1.2, res_max * 1.2], row=3, col=1)

fig.update_layout(height=900, showlegend=False, template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

# --- Explanation ---
st.info(
    r"""
**Visual Guide:**
* **Top Plot:** The signal $x$ is fixed in time.
* **Middle Plot:** The kernel $h$ is flipped ($h(-\tau)$) and shifts ($+t$) as you move the slider. 
* **Bottom Plot:** The green line shows the cumulative area of the overlap between the top two plots. Notice that the peak occurs when the most 'similar' parts of the waves overlap!
"""
)

st.markdown(
    r"""
## Understanding the Convolution Integral
The animation above visualizes the mathematical process of **convolution**. In signal processing and machine learning, convolution is a way to combine two functions to see how the shape of one is modified by the other.

### 1. The Formula
For two continuous functions  and, the convolution  is defined as:
$$(x * h)(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau$$
In our discrete simulation this becomes a summation:
$$(x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] h[n - k]$$

### 2. The "Flip, Shift, Multiply, Sum" Algorithm
If the formula looks intimidating, just remember the four steps you performed with the slider:

1. **Flip:** We take the function  and flip it horizontally to get . This is why you see the "Sawtooth" or "Sine" kernel appear backwards in the middle plot.
2. **Shift:** We slide the flipped function by a time offset . This is the  part of the equation controlled by your slider.
3. **Multiply:** At every point where the two functions overlap, we multiply their values together.
4. **Sum (Integrate):** We calculate the total area under that product curve. This single value becomes the data point for the result plot at time .

### 3. Why do we flip the kernel?

The 'flip' is what distinguishes convolution from cross-correlation. While correlation measures similarity (how well two patterns line up), convolution measures impact. By flipping the kernel, we align the signal's history with the system's response—ensuring that an input at 'time zero' affects the future, not the past.
* In linear systems, this correctly represents the impulse response. If an event happens at $t = 0$, its effect should spread into the future, not the past. 
Flipping the kernel aligns the "history" of the signal with the "response" of the filter.

### 4. What is this used for?

* **Smoothing & Filtering:** Using a "Triangle" or "Boxcar" kernel acts as a low-pass filter, removing high-frequency noise from a signal.
* **Feature Detection:** As you saw with the **Full Sine Wave**, the output is largest when the two functions "align." In Machine Learning, Convolutional Neural Networks (CNNs) use this to scan images for specific patterns like edges, circles, or textures.
* **Signal Analysis:** It tells us how a system (like a speaker or a sensor) will distort an input signal.
    """
)
