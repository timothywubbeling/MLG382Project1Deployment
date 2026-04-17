# IMPORTS
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import joblib

import urllib.request
from pathlib import Path


# =========================
# ARTIFACT SETUP
# =========================
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

ARTIFACTS = {
    "xgboost_model.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/xgboost_model.pkl",
    "scaler.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/scaler.pkl",
    "label_encoder.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/label_encoder.pkl",
    "feature_names.csv": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/feature_names.csv",
}


def download_if_missing(filename, url):
    path = ARTIFACTS_DIR / filename
    if not path.exists():
        urllib.request.urlretrieve(url, path)


for f, u in ARTIFACTS.items():
    download_if_missing(f, u)


# =========================
# LOAD MODEL
# =========================
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

        # ✅ XGBoost-safe SHAP
        _explainer = shap.Explainer(_model)

    return _model, _scaler, _le, _explainer


# =========================
# UI CONFIG
# =========================
RISK_CONFIG = {
    "No Diabetes":  {"color": "#28a745", "bg": "#d4edda", "icon": "✅", "desc": "No indicators of diabetes detected."},
    "Pre-Diabetes": {"color": "#fd7e14", "bg": "#fff3cd", "icon": "⚠️", "desc": "Blood sugar is elevated. Lifestyle changes are recommended."},
    "Gestational":  {"color": "#6f42c1", "bg": "#e8d5f5", "icon": "🤰", "desc": "Limited training data — interpret cautiously."},
    "Type 1":       {"color": "#dc3545", "bg": "#f8d7da", "icon": "🔴", "desc": "Limited training data — interpret cautiously."},
    "Type 2":       {"color": "#c82333", "bg": "#f8d7da", "icon": "🔴", "desc": "Medical review advised."},
}


def labeled_input(label, component):
    return html.Div([
        html.Label(label, style={
            "fontSize": "0.8rem",
            "fontWeight": "600",
            "color": "#6c757d",
            "marginBottom": "4px"
        }),
        component
    ], style={"marginBottom": "12px"})


CARD = {
    "background": "white",
    "padding": "24px",
    "borderRadius": "16px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
    "marginBottom": "24px"
}

SECTION_TITLE = {
    "fontSize": "0.7rem",
    "fontWeight": "700",
    "color": "#adb5bd",
    "marginBottom": "16px"
}


# =========================
# APP
# =========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server


# =========================
# LAYOUT (UNCHANGED)
# =========================
app.layout = html.Div(style={"backgroundColor": "#f7f8fc"}, children=[

    html.Div(style={
        "background": "linear-gradient(135deg, #1a73e8, #0d47a1)",
        "padding": "20px"
    }, children=[
        html.H3("Diabetes Risk Decision Support", style={"color": "white"}),
        html.P("XGBoost Classifier · 91.5% Accuracy", style={"color": "white"})
    ]),

    dbc.Container([

        html.Div(style=CARD, children=[
            html.P("Patient Demographics", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("Age", dbc.Input(id="age", type="number")), md=4),
                dbc.Col(labeled_input("Gender", dcc.Dropdown(
                    id="gender",
                    options=["Male", "Female", "Other"]
                )), md=4),
                dbc.Col(labeled_input("Ethnicity", dcc.Dropdown(
                    id="ethnicity",
                    options=["Black", "White", "Asian", "Hispanic", "Other"]
                )), md=4),
            ])
        ]),

        html.Div(style=CARD, children=[
            html.P("Lifestyle", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("Activity", dbc.Input(id="activity", type="number")), md=4),
                dbc.Col(labeled_input("Alcohol", dbc.Input(id="alcohol", type="number")), md=4),
                dbc.Col(labeled_input("Sleep", dbc.Input(id="sleep", type="number")), md=4),
            ])
        ]),

        html.Div(style=CARD, children=[
            html.P("Medical", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("BMI", dbc.Input(id="bmi", type="number")), md=3),
                dbc.Col(labeled_input("Glucose", dbc.Input(id="glucose", type="number")), md=3),
                dbc.Col(labeled_input("HbA1c", dbc.Input(id="hba1c", type="number")), md=3),
                dbc.Col(labeled_input("BP", dbc.Input(id="bp", type="number")), md=3),
            ])
        ]),

        dbc.Button("Predict", id="predict-btn", color="primary"),

        html.Div(id="result-card"),
        dcc.Graph(id="shap-plot")

    ])
])


# =========================
# CALLBACK (FULL LOGIC)
# =========================
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
        return dbc.Alert("Fill required fields", color="warning"), {}

    try:
        data = FEATURE_MEANS.copy()

        data["Column1"] = age
        data["bmi"] = bmi
        data["glucose_fasting"] = glucose

        data["physical_activity_minutes_per_week"] = activity or data["physical_activity_minutes_per_week"]
        data["alcohol_consumption_per_week"] = alcohol or data["alcohol_consumption_per_week"]
        data["sleep_hours_per_day"] = sleep or data["sleep_hours_per_day"]
        data["hba1c"] = hba1c or data["hba1c"]
        data["systolic_bp"] = bp or data["systolic_bp"]

        data["gender_Male"] = int(gender == "Male") if gender else 0
        data["gender_Other"] = int(gender == "Other") if gender else 0

        data["ethnicity_Black"] = int(ethnicity == "Black") if ethnicity else 0
        data["ethnicity_Hispanic"] = int(ethnicity == "Hispanic") if ethnicity else 0
        data["ethnicity_White"] = int(ethnicity == "White") if ethnicity else 0
        data["ethnicity_Other"] = int(ethnicity == "Other") if ethnicity else 0

        X = pd.DataFrame([data], columns=MODEL_FEATURES)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=MODEL_FEATURES)

        prediction = model.predict(X_scaled)[0]
        label = le.inverse_transform([prediction])[0]
        proba = model.predict_proba(X_scaled)[0]

        cfg = RISK_CONFIG.get(label)

        result = html.Div([
            html.H4(f"{cfg['icon']} {label}"),
            html.P(cfg["desc"])
        ])

        # SHAP FIX
        shap_values = explainer(X_scaled)
        shap_vals = shap_values.values[0]

        if shap_vals.ndim == 2:
            shap_vals = shap_vals[:, prediction]

        shap_df = pd.DataFrame({
            "Feature": MODEL_FEATURES,
            "Impact": shap_vals
        }).sort_values(by="Impact", key=abs, ascending=False).head(10)

        fig = px.bar(shap_df, x="Impact", y="Feature", orientation="h")

        return result, fig

    except Exception as e:
        return dbc.Alert(str(e), color="danger"), {}


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
