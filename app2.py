"""
app.py — NEO Hazard Prediction (Streamlit)
--------------------------------------------
Loads the ANN trained by train_model.py (see artifacts/) and lets a user
predict whether a Near-Earth Object is hazardous, either one at a time
through a form, or in bulk via CSV upload.

Run with:
    streamlit run app.py
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEO Hazard Prediction",
    page_icon="☄️",
    layout="wide",
)

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "ann_model.keras")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "config.json")

RF_METRICS = {
    "Accuracy": 0.8444, "Precision": 0.3610, "Recall": 0.7787,
    "F1 Score": 0.4933, "ROC-AUC": 0.9183,
}

CARD_CSS = """
<style>
.metric-card {
    background: linear-gradient(145deg, #1e2530 0%, #161b22 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 16px 14px 16px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.metric-card .label {
    font-size: 0.80rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #9aa4b2;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0f2f6;
    line-height: 1.1;
}
.metric-card .delta {
    margin-top: 6px;
    font-size: 0.85rem;
    font-weight: 600;
}
.delta-up   { color: #2ecc71; }
.delta-down { color: #e74c3c; }
.delta-flat { color: #9aa4b2; }
</style>
"""


# ── Cached loaders ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, scaler, config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Avg_diameter"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2
    df["Diameter_range"] = df["est_diameter_max"] - df["est_diameter_min"]
    df["Velocity_distance_ratio"] = df["relative_velocity"] / df["miss_distance"]
    df["log_velocity"] = np.log1p(df["relative_velocity"])
    df["log_miss_distance"] = np.log1p(df["miss_distance"])
    return df


def predict(df_raw: pd.DataFrame, model, scaler, config):
    df_feat = engineer_features(df_raw)
    df_feat = df_feat[config["feature_order"]]
    X_scaled = scaler.transform(df_feat)
    probs = model.predict(X_scaled, verbose=0).flatten()
    preds = (probs >= config["best_threshold"]).astype(int)
    return probs, preds


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("☄️ NEO Hazard Predictor")
st.sidebar.markdown(
    "Predicts whether a Near-Earth Object is **hazardous** using an "
    "Artificial Neural Network trained on NASA NEO data."
)
page = st.sidebar.radio("Navigate", ["Single Prediction", "Batch Prediction (CSV)", "Model Info"])

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(CONFIG_PATH)):
    st.error(
        "No trained model found in `artifacts/`.\n\n"
        "Run `python train_model.py` first (with `sprint1.csv` in the same folder) "
        "to generate `artifacts/ann_model.keras`, `scaler.pkl`, and `config.json`, "
        "then relaunch this app."
    )
    st.stop()

model, scaler, config = load_artifacts()
threshold = config["best_threshold"]
test_metrics = config["test_metrics"]

# ── Page: Single Prediction ─────────────────────────────────────────
if page == "Single Prediction":
    st.title("☄️ NEO Hazard Prediction")
    st.caption("Enter an object's orbital parameters to estimate hazard risk.")

    col1, col2 = st.columns(2)
    with col1:
        est_diameter_min = st.number_input(
            "Estimated diameter — min (km)", min_value=0.0, value=0.20, step=0.01, format="%.4f")
        est_diameter_max = st.number_input(
            "Estimated diameter — max (km)", min_value=0.0, value=0.45, step=0.01, format="%.4f")
        relative_velocity = st.number_input(
            "Relative velocity (km/h)", min_value=0.0, value=40000.0, step=1000.0)
    with col2:
        miss_distance = st.number_input(
            "Miss distance (km)", min_value=1.0, value=45000000.0, step=100000.0, format="%.0f")
        absolute_magnitude = st.number_input(
            "Absolute magnitude (H)", min_value=0.0, value=20.0, step=0.1)

    if est_diameter_max < est_diameter_min:
        st.warning("Max diameter is smaller than min diameter — please check your inputs.")

    if st.button("Predict Hazard", type="primary"):
        input_df = pd.DataFrame([{
            "est_diameter_min": est_diameter_min,
            "est_diameter_max": est_diameter_max,
            "relative_velocity": relative_velocity,
            "miss_distance": miss_distance,
            "absolute_magnitude": absolute_magnitude,
        }])
        probs, preds = predict(input_df, model, scaler, config)
        prob = float(probs[0])
        pred = int(preds[0])

        c1, c2 = st.columns([1, 1])
        with c1:
            if pred == 1:
                st.error(f"⚠️ **HAZARDOUS** — predicted probability: {prob:.1%}")
            else:
                st.success(f"✅ **NOT hazardous** — predicted probability: {prob:.1%}")
            st.caption(f"Decision threshold used: {threshold:.2f}")

            eng = engineer_features(input_df).iloc[0]
            st.markdown("**Derived features**")
            st.dataframe(
                pd.DataFrame({
                    "feature": ["Avg diameter", "Diameter range", "Velocity/distance ratio",
                                "log(velocity)", "log(miss distance)"],
                    "value": [round(eng["Avg_diameter"], 4), round(eng["Diameter_range"], 4),
                              round(eng["Velocity_distance_ratio"], 8),
                              round(eng["log_velocity"], 3), round(eng["log_miss_distance"], 3)],
                }),
                hide_index=True, use_container_width=True,
            )

        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "Hazard Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if pred == 1 else "#2ecc71"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#d4edda"},
                        {"range": [threshold * 100, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "orange", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

# ── Page: Batch Prediction ──────────────────────────────────────────
elif page == "Batch Prediction (CSV)":
    st.title("📄 Batch Prediction")
    st.markdown(
        "Upload a CSV with columns: `est_diameter_min`, `est_diameter_max`, "
        "`relative_velocity`, `miss_distance`, `absolute_magnitude`."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            if "Unnamed: 0" in raw_df.columns:
                raw_df = raw_df.drop(columns=["Unnamed: 0"])
            required_cols = ["est_diameter_min", "est_diameter_max",
                              "relative_velocity", "miss_distance", "absolute_magnitude"]
            missing = [c for c in required_cols if c not in raw_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                probs, preds = predict(raw_df[required_cols], model, scaler, config)
                out_df = raw_df.copy()
                out_df["hazard_probability"] = probs
                out_df["predicted_hazardous"] = preds
                st.success(f"Scored {len(out_df):,} objects.")
                st.dataframe(out_df, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    out_df.to_csv(index=False).encode("utf-8"),
                    file_name="neo_predictions.csv",
                    mime="text/csv",
                )
                st.bar_chart(out_df["predicted_hazardous"].value_counts())
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ── Page: Model Info ────────────────────────────────────────────────
else:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("📊 Model Performance Dashboard")
    st.markdown(
        "A 3-hidden-layer **Artificial Neural Network** (Dense + BatchNorm + "
        "Dropout, sigmoid output) trained on NASA Near-Earth Object data, "
        "with class-weighting for the ~9:1 hazard imbalance and F1-optimized "
        "threshold tuning."
    )

    ann_vals = {
        "Accuracy": test_metrics["accuracy"], "Precision": test_metrics["precision"],
        "Recall": test_metrics["recall"], "F1 Score": test_metrics["f1"],
        "ROC-AUC": test_metrics["roc_auc"],
    }

    # ── Metric cards (ANN, with delta vs. Random Forest baseline) ──────
    st.subheader("ANN — Test Set Metrics")
    metric_cols = st.columns(5)
    icons = {"Accuracy": "🎯", "Precision": "🔍", "Recall": "🚨",
             "F1 Score": "⚖️", "ROC-AUC": "📈"}
    for col, name in zip(metric_cols, ann_vals.keys()):
        val = ann_vals[name]
        base = RF_METRICS[name]
        diff = val - base
        cls = "delta-up" if diff > 0.0005 else ("delta-down" if diff < -0.0005 else "delta-flat")
        arrow = "▲" if diff > 0.0005 else ("▼" if diff < -0.0005 else "▶")
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{icons[name]} {name}</div>
                    <div class="value">{val:.3f}</div>
                    <div class="delta {cls}">{arrow} {diff:+.3f} vs RF</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar + grouped bar side by side ────────────────────────────────
    r1, r2 = st.columns([1, 1.2])
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    with r1:
        st.subheader("ANN vs. Random Forest")
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=[RF_METRICS[m] for m in metric_names] + [RF_METRICS[metric_names[0]]],
            theta=metric_names + [metric_names[0]],
            fill="toself", name="Random Forest",
            line=dict(color="#3498db"), opacity=0.55,
        ))
        radar.add_trace(go.Scatterpolar(
            r=[ann_vals[m] for m in metric_names] + [ann_vals[metric_names[0]]],
            theta=metric_names + [metric_names[0]],
            fill="toself", name="ANN",
            line=dict(color="#e74c3c"), opacity=0.55,
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=30, r=30),
            height=420,
        )
        st.plotly_chart(radar, use_container_width=True)

    with r2:
        st.subheader("Metric-by-Metric Comparison")
        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=metric_names, y=[RF_METRICS[m] for m in metric_names],
            name="Random Forest", marker_color="#3498db",
            text=[f"{RF_METRICS[m]:.3f}" for m in metric_names], textposition="outside",
        ))
        bar.add_trace(go.Bar(
            x=metric_names, y=[ann_vals[m] for m in metric_names],
            name="ANN", marker_color="#e74c3c",
            text=[f"{ann_vals[m]:.3f}" for m in metric_names], textposition="outside",
        ))
        bar.update_layout(
            barmode="group", yaxis=dict(range=[0, 1.15], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=40, b=20, l=20, r=20),
            height=420,
        )
        st.plotly_chart(bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture + raw table ────────────────────────────────────────
    a1, a2 = st.columns([1, 1])
    with a1:
        st.subheader("🧠 Architecture (best config)")
        bc = config["best_config"]
        st.markdown(
            f"""
            | Parameter | Value |
            |---|---|
            | Hidden units | {bc['units_1']} → {bc['units_2']} → {bc['units_3']} |
            | Dropout rate | {bc['dropout_rate']} |
            | Learning rate | {bc['lr']} |
            | L2 regularization | {bc['l2']} |
            | Decision threshold | {threshold:.2f} (F1-optimized) |
            """
        )
    with a2:
        st.subheader("📋 Raw Comparison Table")
        comp_df = pd.DataFrame([
            {"Model": "Random Forest (baseline)", **RF_METRICS},
            {"Model": "ANN (this app)", **ann_vals},
        ]).set_index("Model")
        st.dataframe(comp_df.style.format("{:.4f}").background_gradient(
            cmap="RdYlGn", axis=0), use_container_width=True)