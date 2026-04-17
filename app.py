
# ===============================
# IMPORTS
# ===============================
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib

import os
import urllib.request
from pathlib import Path

# ===============================
# ARTIFACTS
# ===============================
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

ARTIFACTS = {
    "random_forest_model.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/xgboost_model.pkl",
    "scaler.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/scaler.pkl",
    "label_encoder.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/label_encoder.pkl",
    "feature_names.csv": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/feature_names.csv",
}

def download_if_missing(filename: str, url: str):
    local_path = ARTIFACTS_DIR / filename
    if local_path.exists():
        print(f"✅ Found local artifact: {filename}")
        return
    print(f"⬇️ Downloading {filename}")
    urllib.request.urlretrieve(url, local_path)

for filename, url in ARTIFACTS.items():
    download_if_missing(filename, url)

# ===============================
# LOAD FEATURE NAMES
# ===============================
MODEL_FEATURES = pd.read_csv(
    ARTIFACTS_DIR / "feature_names.csv",
    header=None
)[0].tolist()

FEATURE_MEANS = {f: 0.0 for f in MODEL_FEATURES}

# ===============================
# MODEL CACHE
# ===============================
_model = None
_scaler = None
_le = None
_explainer = None

def load_model():
    global _model, _scaler, _le, _explainer

    if _model is None:
        _model = joblib.load(ARTIFACTS_DIR / "random_forest_model.pkl")
        _scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        _le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")

        # ✅ ADJUSTMENT: model-agnostic SHAP (SAFE for pickled XGBoost)
        _explainer = shap.Explainer(
            _model.predict_proba,
            feature_names=MODEL_FEATURES
        )

    return _model, _scaler, _le, _explainer

# ===============================
# RISK CONFIG
# ===============================
RISK_CONFIG = {
    "No Diabetes":  {"color": "#28a745", "bg": "#d4edda", "icon": "✅", "desc": "No indicators of diabetes detected."},
    "Pre-Diabetes": {"color": "#fd7e14", "bg": "#fff3cd", "icon": "⚠️", "desc": "Blood sugar is elevated. Lifestyle changes are recommended."},
    "Gestational":  {"color": "#6f42c1", "bg": "#e8d5f5", "icon": "🤰", "desc": "Gestational diabetes pattern detected."},
    "Type 1":       {"color": "#dc3545", "bg": "#f8d7da", "icon": "🔴", "desc": "Type 1 diabetes indicators detected."},
    "Type 2":       {"color": "#c82333", "bg": "#f8d7da", "icon": "🔴", "desc": "Type 2 diabetes indicators detected."},
}

# ===============================
# DASH APP
# ===============================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
server = app.server

# ===============================
# LAYOUT (UNCHANGED)
# ===============================
app.layout = dbc.Container([
    dbc.Button("Assess Diabetes Risk", id="predict-btn", color="primary"),
    html.Div(id="result-card"),
    dcc.Graph(id="shap-plot")
], fluid=True)

# ===============================
# CALLBACK
# ===============================
@app.callback(
    Output("result-card", "children"),
    Output("shap-plot", "figure"),
    Input("predict-btn", "n_clicks")
)
def predict(n_clicks):

    if not n_clicks:
        return "", {}

    model, scaler, le, explainer = load_model()

    data = FEATURE_MEANS.copy()
    X_raw = pd.DataFrame([data], columns=MODEL_FEATURES)
    X_scaled = scaler.transform(X_raw)
    X_np = X_scaled  # ✅ ADJUSTMENT: NumPy-safe for XGBoost

    # ===============================
    # PREDICTION
    # ===============================
    prediction = int(model.predict(X_np)[0])
    proba = model.predict_proba(X_np)[0]
    label = le.inverse_transform([prediction])[0]
    confidence = round(proba[prediction] * 100, 1)

    result_card = dbc.Alert(
        f"{label} — confidence {confidence}%",
        color="primary"
    )

    # ===============================
    # SHAP (SAFE FOR PICKLED XGBOOST)
    # ===============================
    shap_values = explainer(X_np)

    # ✅ ADJUSTMENT: correct class-specific SHAP values
    shap_vals = shap_values.values[0, :, prediction]

    shap_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Impact": shap_vals
    }).sort_values(by="Impact", key=abs, ascending=False).head(10)

    fig = px.bar(
        shap_df,
        x="Impact",
        y="Feature",
        orientation="h",
        template="plotly_white",
        title=f"Feature Impact on '{label}'"
    )
    fig.add_vline(x=0, line_color="#ccc")

    return result_card, fig

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run_server(debug=True)
