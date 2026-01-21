import streamlit as st

# Define the logical flow of the course
pages = {
    "1. Foundations": [
        st.Page("transform.py", title="Function Approximations"),
        st.Page("learning_methods.py", title="Supervised vs Unsupervised"),
        st.Page("gradient_descent.py", title="Optimization (Gradient Descent)"),
    ],
    "2. Neural Building Blocks": [
        st.Page("hidden_layers.py", title="Hidden Layers (MLP)"),
        st.Page("activation_fn.py", title="Activation Functions"),
        st.Page("conv_1d.py", title="1D Convolutions (Signals)"),
        st.Page("conv_2d.py", title="2D Convolutions (Images)"),
        st.Page("architectures.py", title="Common Architectures"),
    ],
    "3. Practical Labs": [
        st.Page("build_model.py", title="Lab 1: Build a Model"),
        st.Page("regression.py", title="Lab 2: Model Capacity"),
        st.Page("augmentation.py", title="Lab 3: Data Augmentation"),
        st.Page("dynamics.py", title="Lab 4: Overfitting Simulator"),
        st.Page("metrics.py", title="Lab 5: Evaluation Metrics"),
    ],
    "4. Tuning & Production": [
        st.Page("tuning.py", title="Hyperparameter Tuning (Basic)"),
        st.Page("tuning_torch.py", title="Hyperparameter Tuning (Torch)"),
        st.Page("inference.py", title="Live Inference"),
    ],
}

pg = st.navigation(pages)
pg.run()
