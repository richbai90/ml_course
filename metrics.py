import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from utils import ask_ai, lightbulb_callout

st.set_page_config(layout="wide")
st.title("Lab: Evaluation Metrics")
lightbulb_callout('Big Picture', r'''Performance quantification beyond scalar accuracy. We examine metrics for class-imbalanced datasets, including Precision, Recall, and F1-Scores. For geometric tasks, we introduce IoU (Intersection over Union) to
                  quantify the topological overlap between predicted and ground-truth bounding boxes.''')
st.markdown("""
**How do we know if our model is "good"?** In Computational Imaging, **Accuracy** is often misleading. 
* If 99% of your pixels are background and 1% are a tumor, a model that predicts "background" for everything has 99% accuracy but is useless.
* We need metrics that measure **overlaps**, **sensitivity**, and **precision**.
""")

# --- Helper: IoU Calculation ---
def calculate_iou(box1, box2):
    """
    Box format: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0: return 0
    return intersection_area / union_area

# --- TAB SELECTION ---
tab1, tab2 = st.tabs(["1. Classification (Confusion Matrix & ROC)", "2. Detection (IoU)"])

# ==========================================
# TAB 1: CLASSIFICATION
# ==========================================
with tab1:
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("The Decision Threshold")
        st.write("A model outputs a **probability** (0.0 to 1.0). We must choose a cutoff to decide 'Yes' or 'No'.")
        
        # 1. Generate Synthetic Data
        np.random.seed(42)
        n_samples = 1000
        # Create two distributions: "Healthy" (0) and "Disease" (1)
        # Disease scores are generally higher, but overlap exists
        y_true = np.concatenate([np.zeros(900), np.ones(100)]) # 10% imbalance
        
        # Probabilities
        noise = np.random.normal(0, 0.15, n_samples)
        # Shift scores: healthy around 0.3, disease around 0.7
        y_scores = np.concatenate([np.random.normal(0.3, 0.15, 900), np.random.normal(0.7, 0.15, 100)])
        y_scores = np.clip(y_scores, 0, 1) # bound between 0 and 1
        
        threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, step=0.01, 
                              help="Probabilities above this value are predicted as Positive (Disease).")
        
        st.markdown("---")
        st.markdown("**Metric Definitions:**")
        st.markdown("**Precision:** Of all predicted positives, how many are real? (Avoid False Positives)")
        st.markdown("**Recall (Sensitivity):** Of all real positives, how many did we find? (Avoid Missing Disease)")
        st.markdown("**F1 Score:** The Harmonic Mean of Precision and Recall. It penalizes extreme values, ensuring a balance between finding all targets and avoiding false Positives.")
        
    with col_viz:
        # Calculate Logic
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion Matrix
        #      Pred 0   Pred 1
        # True 0   TN       FP
        # True 1   FN       TP
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Metrics
        accuracy = (tp + tn) / n_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # TPR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # --- Row 1: Metrics & Matrix ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.2%}")
        m2.metric("Precision", f"{precision:.2%}")
        m3.metric("Recall", f"{recall:.2%}")
        m4.metric("F1 Score", f"{f1:.2f}")
        
        # Plot Confusion Matrix
        cm_data = [[tn, fp], [fn, tp]]
        fig_cm = px.imshow(cm_data, 
                           text_auto=True, 
                           color_continuous_scale="Blues",
                           labels=dict(x="Predicted Label", y="True Label", color="Count"),
                           x=["Negative (Healthy)", "Positive (Disease)"],
                           y=["Negative (Healthy)", "Positive (Disease)"])
        fig_cm.update_layout(title="Confusion Matrix", height=300)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # --- Row 2: ROC Curve ---
        st.subheader("ROC Curve (Receiver Operating Characteristic)")
        st.caption("Shows the trade-off between Sensitivity (Recall) and False Positives (FPR) for ALL thresholds.")
        
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr_vals, tpr_vals)
        
        fig_roc = go.Figure()
        
        # The Curve
        fig_roc.add_trace(go.Scatter(x=fpr_vals, y=tpr_vals, name=f"ROC (AUC={roc_auc:.2f})", mode='lines', line=dict(color='blue')))
        
        # The Current Threshold Point
        fig_roc.add_trace(go.Scatter(x=[fpr], y=[recall], name=f"Threshold {threshold}", 
                                     mode='markers', marker=dict(size=12, color='red', symbol='x')))
        
        # Random Guess Line
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Guess", mode='lines', line=dict(dash='dash', color='gray')))
        
        fig_roc.update_layout(xaxis_title="False Positive Rate (1 - Specificity)", 
                              yaxis_title="True Positive Rate (Recall)",
                              height=400, showlegend=True)
        st.plotly_chart(fig_roc, use_container_width=True)

# ==========================================
# TAB 2: DETECTION (IoU)
# ==========================================
with tab2:
    st.subheader("Intersection over Union (IoU)")
    st.markdown("""
    In Object Detection (finding a car, a cell, or a face), "Accuracy" is hard to define because the box is never *exactly* on the pixel.
    
    We use **IoU**:
    $$ IoU = \\frac{\\text{Area of Overlap}}{\\text{Area of Union}} $$
    
    * **IoU > 0.5** is typically considered a "hit" or "True Positive".
    """)
    
    col_iou_ctrl, col_iou_viz = st.columns([1, 2])
    
    # Ground Truth Box (Fixed for simplicity)
    gt_box = [100, 100, 300, 300] # x_min, y_min, x_max, y_max
    
    with col_iou_ctrl:
        st.write("Adjust the **Predicted** Box:")
        p_x = st.slider("X Position", 0, 400, 120)
        p_y = st.slider("Y Position", 0, 400, 120)
        p_w = st.slider("Width", 50, 300, 180)
        p_h = st.slider("Height", 50, 300, 180)
        
        pred_box = [p_x, p_y, p_x + p_w, p_y + p_h]
        
        current_iou = calculate_iou(gt_box, pred_box)
        
        st.metric("IoU Score", f"{current_iou:.3f}")
        
        if current_iou > 0.5:
            st.success("Valid Detection (IoU > 0.5)")
        else:
            st.error("Miss / Poor Localization (IoU < 0.5)")
            
    with col_iou_viz:
        # Visualizing Boxes
        fig_iou = go.Figure()
        
        # Set canvas size
        fig_iou.update_layout(
            xaxis=dict(range=[0, 500], showgrid=False),
            yaxis=dict(range=[500, 0], showgrid=False), # Inverted Y for image coords
            height=500,
            width=500,
            title="Bounding Box Visualization"
        )
        
        # Ground Truth (Green)
        fig_iou.add_shape(type="rect",
            x0=gt_box[0], y0=gt_box[1], x1=gt_box[2], y1=gt_box[3],
            line=dict(color="Green", width=3),
            fillcolor="rgba(0, 255, 0, 0.2)",
        )
        fig_iou.add_trace(go.Scatter(x=[gt_box[0]], y=[gt_box[1]], text=["Ground Truth"], mode="text", textposition="top right", showlegend=False))

        # Prediction (Red)
        fig_iou.add_shape(type="rect",
            x0=pred_box[0], y0=pred_box[1], x1=pred_box[2], y1=pred_box[3],
            line=dict(color="Red", width=3, dash="dot"),
            fillcolor="rgba(255, 0, 0, 0.2)",
        )
        fig_iou.add_trace(go.Scatter(x=[pred_box[0]], y=[pred_box[1]], text=["Prediction"], mode="text", textposition="top left", showlegend=False))
        
        st.plotly_chart(fig_iou)

st.divider()
ask_ai()
