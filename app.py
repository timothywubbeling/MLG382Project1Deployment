# IMPORTS
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


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

ARTIFACTS = {
    "xgboost_model.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/xgboost_model.pkl",
    "scaler.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/scaler.pkl",
    "label_encoder.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/label_encoder.pkl",
    "feature_names.csv": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/feature_names.csv",
}


def download_if_missing(filename: str, url: str):
    local_path = ARTIFACTS_DIR / filename

    if local_path.exists():
        print(f"✅ Found local artifact: {filename}")
        return

    print(f"⬇️ Downloading {filename} from {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"✅ Successfully downloaded {filename}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download {filename}: {e}")


# DOWNLOAD ARTIFACTS
for filename, url in ARTIFACTS.items():
    download_if_missing(filename, url)


# LOAD MODEL
_model = None
_scaler = None
_le = None
_explainer = None

def load_model():
    global _model, _scaler, _le, _explainer

    if _model is None:
        _model = joblib.load(ARTIFACTS_DIR / "xgboost_model.pkl")
        _scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        _le = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")

        # ✅ XGBoost-compatible SHAP
        _explainer = shap.Explainer(_model)

    return _model, _scaler, _le, _explainer


# RISK CONFIG
RISK_CONFIG = {
    "No Diabetes":  {"color": "#28a745", "bg": "#d4edda", "icon": "✅", "desc": "No indicators of diabetes detected."},
    "Pre-Diabetes": {"color": "#fd7e14", "bg": "#fff3cd", "icon": "⚠️", "desc": "Blood sugar is elevated. Lifestyle changes are recommended."},
    "Gestational":  {"color": "#6f42c1", "bg": "#e8d5f5", "icon": "🤰", "desc": "Gestational diabetes pattern detected. Note: limited training data for this class — treat with caution."},
    "Type 1":       {"color": "#dc3545", "bg": "#f8d7da", "icon": "🔴", "desc": "Type 1 diabetes indicators detected. Note: limited training data for this class — treat with caution."},
    "Type 2":       {"color": "#c82333", "bg": "#f8d7da", "icon": "🔴", "desc": "Type 2 diabetes indicators detected. Medical review advised."},
}


def labeled_input(label, input_component):
    return html.Div([
        html.Label(label, style={"fontSize": "0.8rem", "fontWeight": "600",
                                 "color": "#6c757d", "marginBottom": "4px",
                                 "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        input_component
    ], style={"marginBottom": "12px"})


CARD = {
    "background": "white",
    "padding": "24px 28px",
    "borderRadius": "16px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
    "marginBottom": "24px",
    "border": "1px solid #f0f0f0"
}

SECTION_TITLE = {
    "fontSize": "0.7rem",
    "fontWeight": "700",
    "color": "#adb5bd",
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "marginBottom": "16px"
}


# DASH APP
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
server = app.server


# LAYOUT (UNCHANGED)
app.layout = html.Div(style={"backgroundColor": "#f7f8fc", "minHeight": "100vh", "fontFamily": "'Segoe UI', sans-serif"}, children=[

    # NAVBAR
    html.Div(style={
        "background": "linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%)",
        "padding": "18px 40px",
        "display": "flex",
        "alignItems": "center",
        "gap": "14px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"
    }, children=[
        html.Span("🩺", style={"fontSize": "1.8rem"}),
        html.Div([
            html.H4("Diabetes Risk Decision Support", style={"margin": 0, "color": "white", "fontWeight": "700"}),
            html.P("BC Analytics · XGBoost Classifier · 91.5% Accuracy",
                   style={"margin": 0, "color": "rgba(255,255,255,0.7)", "fontSize": "0.8rem"})
        ])
    ]),

    # MAIN CONTENT
    dbc.Container(fluid=False, style={"maxWidth": "960px", "paddingTop": "32px", "paddingBottom": "48px"}, children=[

        # DEMOGRAPHICS
        html.Div(style=CARD, children=[
            html.P("Patient Demographics", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("Age", dbc.Input(id="age", type="number", placeholder="e.g. 45", min=1, max=120)), md=4),
                dbc.Col(labeled_input("Gender", dcc.Dropdown(
                    id="gender",
                    options=[{"label": "Female", "value": "Female"},
                             {"label": "Male",   "value": "Male"},
                             {"label": "Other",  "value": "Other"}],
                    placeholder="Select gender", clearable=False
                )), md=4),
                dbc.Col(labeled_input("Ethnicity", dcc.Dropdown(
                    id="ethnicity",
                    options=[{"label": "Asian",    "value": "Asian"},
                             {"label": "Black",    "value": "Black"},
                             {"label": "Hispanic", "value": "Hispanic"},
                             {"label": "White",    "value": "White"},
                             {"label": "Other",    "value": "Other"}],
                    placeholder="Select ethnicity", clearable=False
                )), md=4),
            ])
        ]),

        # LIFESTYLE
        html.Div(style=CARD, children=[
            html.P("Lifestyle Factors", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("Physical Activity (min/week)",
                    dbc.Input(id="activity", type="number", placeholder="e.g. 150")), md=4),
                dbc.Col(labeled_input("Alcohol (drinks/week)",
                    dbc.Input(id="alcohol", type="number", placeholder="e.g. 2")), md=4),
                dbc.Col(labeled_input("Sleep (hours/day)",
                    dbc.Input(id="sleep", type="number", placeholder="e.g. 7")), md=4),
            ])
        ]),

        # MEDICAL
        html.Div(style=CARD, children=[
            html.P("Medical & Lab Measurements", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("BMI *",
                    dbc.Input(id="bmi", type="number", placeholder="e.g. 27.5")), md=3),
                dbc.Col(labeled_input("Fasting Glucose (mg/dL) *",
                    dbc.Input(id="glucose", type="number", placeholder="e.g. 110")), md=3),
                dbc.Col(labeled_input("HbA1c (%)",
                    dbc.Input(id="hba1c", type="number", placeholder="e.g. 6.5")), md=3),
                dbc.Col(labeled_input("Systolic BP (mmHg)",
                    dbc.Input(id="bp", type="number", placeholder="e.g. 120")), md=3),
            ]),
            html.P("* Required fields", style={"fontSize": "0.75rem", "color": "#adb5bd", "margin": 0})
        ]),

        # BUTTON
        dbc.Button(
            "Assess Diabetes Risk",
            id="predict-btn",
            color="primary",
            size="lg",
            style={
                "width": "100%",
                "padding": "14px",
                "fontSize": "1rem",
                "fontWeight": "600",
                "borderRadius": "12px",
                "background": "linear-gradient(135deg, #1a73e8, #0d47a1)",
                "border": "none",
                "marginBottom": "28px",
                "boxShadow": "0 4px 12px rgba(26,115,232,0.35)"
            }
        ),

        html.Div(id="result-card"),
        dcc.Graph(id="shap-plot", style={"marginTop": "8px"}),

        html.Hr(style={"borderColor": "#e9ecef", "marginTop": "40px"}),
        html.P(
            "This tool is for decision support only and does not constitute medical advice.",
            style={"textAlign": "center", "fontSize": "0.75rem", "color": "#adb5bd"}
        )
    ])
])


# CALLBACK
@app.callback(
    Output("result-card", "children"),
    Output("shap-plot", "figure"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("gender", "value"),
    State("ethnicity", "value"),
    State("activity", "value"),
    State("alcohol", "value"),
    State("sleep", "value"),
    State("bmi", "value"),
    State("glucose", "value"),
    State("hba1c", "value"),
    State("bp", "value"),
)
def predict(n_clicks, age, gender, ethnicity, activity, alcohol, sleep, bmi, glucose, hba1c, bp):

    model, scaler, le, explainer = load_model()

    if not n_clicks:
        return "", {}

    if None in [age, bmi, glucose]:
        return dbc.Alert("Please fill in required fields.", color="warning"), {}

    try:
        data = FEATURE_MEANS.copy()
        data["Column1"] = age
        data["bmi"] = bmi
        data["glucose_fasting"] = glucose

        X_raw = pd.DataFrame([data], columns=MODEL_FEATURES)
        X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=MODEL_FEATURES)

        prediction = model.predict(X_scaled)[0]
        label = le.inverse_transform([prediction])[0]

        shap_values = explainer(X_scaled)
        shap_vals = shap_values.values[0]

        if shap_vals.ndim == 2:
            shap_vals = shap_vals[:, prediction]

        fig = px.bar(x=shap_vals, y=MODEL_FEATURES, orientation="h")

        return html.H4(label), fig

    except Exception as e:
        return dbc.Alert(str(e), color="danger"), {}

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
