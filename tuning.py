import time

import numpy as np
import optuna
import plotly.graph_objects as go
import streamlit as st

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Hyperparameter Tuning with Optuna")
lightbulb_callout('Big Picture', r'''Optimization of non-differentiable hyperparameters. Unlike weights, parameters like learning rate or architecture depth cannot be learned via backpropagation. This module demonstrates Bayesian Optimization (via Optuna)
                  to efficiently search the hyperparameter space for the global minimum.''')
# --- Section 1: The Theory ---
st.markdown(r"""
### 1. What are Hyperparameters?
In our **Gradient Descent** lesson, the model learned the "Weights" ($w$) automatically by minimizing loss. However, there were some settings **we** had to choose before training started, like the **Learning Rate** ($\alpha$).

These "settings you choose" are called **Hyperparameters**.

### 2. Why do they matter?
* **Too Low:** The model learns too slowly (waste of time).
* **Too High:** The model diverges or behaves unstably.
* **Just Right:** The model converges quickly to the best solution.

Finding the "Just Right" values is often done by trial and error, or by using automated tools like **Optuna**.
""")

st.divider()

# --- Section 2: Interactive Optuna Demo ---
st.header("Automated Tuning Demo")
st.markdown(r"""
Imagine a black-box function we want to minimize:
$$f(x, y) = (x - 2)^2 + (y + 3)^2 + x \cdot \sin(y)$$
We don't know the math formula, we just pass in $x$ and $y$ and get a score. Optuna will try to find the $x$ and $y$ that give the lowest score.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    n_trials = st.slider("Number of Trials", 10, 100, 20)
    
    if st.button("Start Tuning Study"):
        
        # 1. Define the Objective Function
        def objective(trial):
            # Optuna suggests values for x and y
            x = trial.suggest_float("x", -10, 10)
            y = trial.suggest_float("y", -10, 10)
            
            # The function to minimize
            score = (x - 2)**2 + (y + 3)**2 + x * np.sin(y)
            return score

        # 2. Create a Study
        study = optuna.create_study(direction="minimize")
        
        # 3. Optimize!
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # We run the loop manually to update the UI
        for i in range(n_trials):
            study.optimize(objective, n_trials=1)
            progress_bar.progress((i + 1) / n_trials)
            status_text.text(f"Trial {i+1}/{n_trials}: Current Best = {study.best_value:.4f}")
            time.sleep(0.05) # Just for visual effect
            
        st.success("Tuning Complete!")
        
        st.write("### Best Parameters Found:")
        st.json(study.best_params)
        st.write(f"**Best Loss:** {study.best_value:.4f}")

        # Store study in session state to persist plot
        st.session_state["study"] = study

with col2:
    if "study" in st.session_state:
        st.subheader("Optimization History")
        
        # Extract data for plotting
        trials = st.session_state["study"].trials
        trial_nums = [t.number for t in trials]
        values = [t.value for t in trials]
        
        # Highlight the best trial
        best_trial = st.session_state["study"].best_trial
        
        fig = go.Figure()
        
        # Plot all trials
        fig.add_trace(go.Scatter(
            x=trial_nums, 
            y=values,
            mode='markers+lines',
            name='Trial Loss',
            line=dict(color='gray', dash='dot'),
            marker=dict(size=8, color='blue', opacity=0.6)
        ))
        
        # Plot best trial
        fig.add_trace(go.Scatter(
            x=[best_trial.number],
            y=[best_trial.value],
            mode='markers',
            name='Best Result',
            marker=dict(size=14, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title="Loss over Time (Lower is Better)",
            xaxis_title="Trial Number",
            yaxis_title="Loss Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Start Tuning Study' to see Optuna in action.")

st.divider()
ask_ai()
