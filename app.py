
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
    "random_forest_model.pkl": "https://huggingface.co/timothywubbeling/ForestDecisionTree/resolve/main/random_forest_model.pkl",
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


## =====================================================
## MODEL BOOTSTRAP (local-first, download-if-missing)
## =====================================================

for filename, url in ARTIFACTS.items():
    download_if_missing(filename, url)

## =====================================================
## LOAD MODEL + FEATURES
## =====================================================

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
        _explainer = shap.TreeExplainer(_model)

    return _model, _scaler, _le, _explainer

# RISK CONFIG — colour + description per class
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


# LAYOUT
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
            html.P("BC Analytics · Random Forest Classifier · 91.5% Accuracy",
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

        # PREDICT BUTTON
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

        # RESULT + CONFIDENCE
        html.Div(id="result-card"),

        # SHAP CHART
        dcc.Graph(id="shap-plot", style={"marginTop": "8px"}),

        # FOOTER
        html.Hr(style={"borderColor": "#e9ecef", "marginTop": "40px"}),
        html.P(
            "This tool is for decision support only and does not constitute medical advice. "
            "Always consult a qualified healthcare professional.",
            style={"textAlign": "center", "fontSize": "0.75rem", "color": "#adb5bd"}
        )
    ])
])


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
def predict(
    n_clicks, age, gender, ethnicity,
    activity, alcohol, sleep,
    bmi, glucose, hba1c, bp
):
    # ✅ FIRST LINE: lazy-load heavy objects
    model, scaler, le, explainer = load_model()

    if not n_clicks:
        return "", {}

    if None in [age, bmi, glucose]:
        alert = dbc.Alert(
            "Please fill in Age, BMI and Fasting Glucose before assessing.",
            color="warning",
            style={"borderRadius": "12px"}
        )
        return alert, {}

    # --- rest of your existing logic continues unchanged ---
    try:
        # BUILD INPUT from training means, then override with user values
        data = FEATURE_MEANS.copy()
        data["Column1"]                            = age
        data["bmi"]                                = bmi
        data["glucose_fasting"]                    = glucose
        data["physical_activity_minutes_per_week"] = activity if activity is not None else data["physical_activity_minutes_per_week"]
        data["alcohol_consumption_per_week"]       = alcohol  if alcohol  is not None else data["alcohol_consumption_per_week"]
        data["sleep_hours_per_day"]                = sleep    if sleep    is not None else data["sleep_hours_per_day"]
        data["hba1c"]                              = hba1c    if hba1c    is not None else data["hba1c"]
        data["systolic_bp"]                        = bp       if bp       is not None else data["systolic_bp"]
        data["gender_Male"]        = int(gender == "Male")     if gender    else 0
        data["gender_Other"]       = int(gender == "Other")    if gender    else 0
        data["ethnicity_Black"]    = int(ethnicity == "Black")    if ethnicity else 0
        data["ethnicity_Hispanic"] = int(ethnicity == "Hispanic") if ethnicity else 0
        data["ethnicity_White"]    = int(ethnicity == "White")    if ethnicity else 0
        data["ethnicity_Other"]    = int(ethnicity == "Other")    if ethnicity else 0

        X_raw    = pd.DataFrame([data], columns=MODEL_FEATURES)
        X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=MODEL_FEATURES)

        # PREDICT + CONFIDENCE
        prediction  = model.predict(X_scaled)[0]
        label       = le.inverse_transform([prediction])[0]
        proba       = model.predict_proba(X_scaled)[0]
        confidence  = round(proba[prediction] * 100, 1)
        cfg         = RISK_CONFIG.get(label, {"color": "#6c757d", "bg": "#f8f9fa", "icon": "❓", "desc": ""})

        # RESULT CARD
        all_classes = le.classes_
        prob_rows = [
            dbc.Col(html.Div([
                html.Div(cls, style={"fontSize": "0.7rem", "color": "#6c757d", "marginBottom": "4px"}),
                dbc.Progress(
                    value=round(proba[i] * 100, 1),
                    label=f"{round(proba[i]*100,1)}%",
                    color="primary" if cls == label else "secondary",
                    style={"height": "10px", "borderRadius": "6px"}
                )
            ]), md=4, style={"marginBottom": "10px"})
            for i, cls in enumerate(all_classes)
        ]

        result_card = html.Div(style={
            **CARD,
            "borderLeft": f"6px solid {cfg['color']}",
            "backgroundColor": cfg["bg"]
        }, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "8px"}, children=[
                html.Span(cfg["icon"], style={"fontSize": "2rem"}),
                html.Div([
                    html.H4(label, style={"margin": 0, "color": cfg["color"], "fontWeight": "700"}),
                    html.P(cfg["desc"], style={"margin": 0, "color": "#495057", "fontSize": "0.9rem"})
                ])
            ]),
            html.Hr(style={"borderColor": cfg["color"], "opacity": "0.3"}),
            html.P("Class Probabilities", style={**SECTION_TITLE, "marginBottom": "10px"}),
            dbc.Row(prob_rows)
        ])

        # SHAP
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals = shap_values[0, :, prediction]
        elif isinstance(shap_values, list):
            shap_vals = shap_values[prediction][0]
        else:
            shap_vals = shap_values[0]

        shap_vals = np.array(shap_vals).flatten()
        shap_df = pd.DataFrame({
            "Feature": MODEL_FEATURES,
            "Impact":  shap_vals
        }).sort_values(by="Impact", key=abs, ascending=False).head(10)

        # Friendly feature name mapping
        name_map = {
            "Column1": "Age", "bmi": "BMI", "glucose_fasting": "Fasting Glucose",
            "hba1c": "HbA1c", "systolic_bp": "Systolic BP",
            "physical_activity_minutes_per_week": "Physical Activity",
            "alcohol_consumption_per_week": "Alcohol Consumption",
            "sleep_hours_per_day": "Sleep Hours", "diabetes_risk_score": "Risk Score",
            "glucose_postprandial": "Postprandial Glucose", "insulin_level": "Insulin Level",
            "waist_to_hip_ratio": "Waist-Hip Ratio",
        }
        shap_df["Feature"] = shap_df["Feature"].apply(lambda x: name_map.get(x, x.replace("_", " ").title()))
        shap_df["Direction"] = shap_df["Impact"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")

        fig = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={"Increases Risk": "#dc3545", "Decreases Risk": "#28a745"},
            title=f"Feature Impact on '{label}' Prediction",
            labels={"Impact": "SHAP Value (impact on model output)", "Feature": ""}
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title_font_size=14,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.add_vline(x=0, line_width=1, line_color="#dee2e6")

        return result_card, fig

    except Exception as e:
        import traceback
        err = dbc.Alert(f"Error: {traceback.format_exc()}", color="danger", style={"borderRadius": "12px", "whiteSpace": "pre-wrap"})
        return err, {}


# RUN
if __name__ == "__main__":
    try:
        app.run(debug=True)
    except AttributeError:
        app.run_server(debug=True)
