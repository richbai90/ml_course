import numpy as np
import optuna
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(layout="wide")
st.title("Tuning PyTorch Models with Optuna")

st.markdown("""
### Why tune Neural Networks?
In Deep Learning, the "architecture" itself is a hyperparameter. 
* **How many layers?** (Too few = underfitting, Too many = overfitting/slow)
* **How many neurons per layer?**
* **Which optimizer?** (SGD, Adam, RMSprop?)

Below, we will use **Optuna** to automatically design a PyTorch Neural Network to fit a noisy function.
""")

# --- 1. Prepare Data (The Problem) ---
# We generate a noisy sine wave: y = sin(x) + noise
torch.manual_seed(42)
X_numpy = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + np.random.normal(0, 0.1, X_numpy.shape)

# Convert to PyTorch Tensors
X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(y_numpy, dtype=torch.float32)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Tuning Configuration")
    n_trials = st.slider("Number of Trials", 5, 50, 10, help="How many different models to try.")
    epochs = st.slider("Epochs per Trial", 50, 500, 100, help="Training iterations per model.")
    
    start_btn = st.button("Start PyTorch Tuning")

    # --- 2. Define the Model Builder ---
    def create_model(trial):
        """
        Optuna suggests the architecture structure here.
        """
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        
        input_dim = 1
        for i in range(n_layers):
            # Optuna suggests neurons for this specific layer
            output_dim = trial.suggest_int(f"n_units_l{i}", 4, 64)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
            
        layers.append(nn.Linear(input_dim, 1)) # Final output layer
        return nn.Sequential(*layers)

    # --- 3. Define the Objective Function ---
    def objective(trial):
        # A. Build Model
        model = create_model(trial)
        
        # B. Choose Optimizer & LR
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)
            
        criterion = nn.MSELoss()
        
        # C. Train Loop (Short)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            # Pruning (Optional: Stop bad trials early)
            trial.report(loss.item(), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return loss.item()

    if start_btn:
        with st.spinner("Optuna is searching for the best neural network..."):
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            
            st.success("Tuning Complete!")
            st.session_state["torch_study"] = study
            st.session_state["best_params"] = study.best_params

            # Re-train best model for plotting
            best_trial = study.best_trial
            st.write(f"**Best Loss:** {best_trial.value:.5f}")
            st.json(study.best_params)

# --- 4. Visualization ---
with col2:
    if "torch_study" in st.session_state:
        st.subheader("Result Visualization")
        
        # Retrain the "Best" model to show its prediction
        best_params = st.session_state["best_params"]
        
        # Reconstruct the best model manually from params
        layers = []
        input_dim = 1
        n_layers = best_params["n_layers"]
        
        for i in range(n_layers):
            units = best_params[f"n_units_l{i}"]
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        layers.append(nn.Linear(input_dim, 1))
        best_model = nn.Sequential(*layers)
        
        # Train it briefly just to get the fit (simulating the best trial state)
        # In a real app, you might save the state_dict, but retraining is easier for a demo
        opt_name = best_params["optimizer"]
        lr = best_params["lr"]
        if opt_name == "Adam":
            opt = optim.Adam(best_model.parameters(), lr=lr)
        else:
            opt = optim.SGD(best_model.parameters(), lr=lr)
            
        loss_fn = nn.MSELoss()
        for _ in range(epochs): # Train for same amount as trial
            opt.zero_grad()
            y_p = best_model(X)
            l = loss_fn(y_p, y)
            l.backward()
            opt.step()
            
        # Predictions
        with torch.no_grad():
            preds = best_model(X).numpy()
            
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_numpy.flatten(), y=y_numpy.flatten(), mode='markers', name='Noisy Data', marker=dict(color='gray', opacity=0.5)))
        fig.add_trace(go.Scatter(x=X_numpy.flatten(), y=preds.flatten(), mode='lines', name='Best Model Fit', line=dict(color='red', width=3)))
        
        fig.update_layout(title="Best Model Performance", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig, use_container_width=True)
        
        # Optuna Plot
        st.write("### Optimization History")
        trials = st.session_state["torch_study"].trials
        t_nums = [t.number for t in trials]
        t_vals = [t.value for t in trials]
        
        fig2 = go.Figure(data=go.Scatter(x=t_nums, y=t_vals, mode='markers+lines', name='Trial Loss'))
        fig2.update_layout(xaxis_title="Trial", yaxis_title="MSE Loss (Lower is Better)")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        # Show placeholder
        st.info("👈 Set parameters and click 'Start PyTorch Tuning' to begin.")
        
        # Static example plot
        fig_static = go.Figure()
        fig_static.add_trace(go.Scatter(x=X_numpy.flatten(), y=y_numpy.flatten(), mode='markers', name='Data', marker=dict(color='gray')))
        fig_static.update_layout(title="Target Dataset: Noisy Sine Wave", height=400)
        st.plotly_chart(fig_static, use_container_width=True)
