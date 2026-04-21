from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

from data_pipeline import build_features_and_target, load_data

MODEL_PATH = Path("models/xgb_bike_model.joblib")
DATA_PATH = Path("data/hour.csv")


@st.cache_resource
def load_model(model_path: Path):
    """Load and cache the trained XGBoost model for fast repeated inference."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data
def load_feature_reference(data_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Build reusable reference values:
    - feature means as a template for non-user-controlled fields
    - scaling statistics for temp/hum/windspeed to match training preprocessing
    """
    raw_df = load_data(data_path)
    X, _ = build_features_and_target(data_path)

    template = X.mean(numeric_only=True).to_dict()

    stats = {
        "temp_mean": float(raw_df["temp"].mean()),
        "temp_std": float(raw_df["temp"].std(ddof=0)),
        "hum_mean": float(raw_df["hum"].mean()),
        "hum_std": float(raw_df["hum"].std(ddof=0)),
        "windspeed_mean": float(raw_df["windspeed"].mean()),
        "windspeed_std": float(raw_df["windspeed"].std(ddof=0)),
    }

    return template, stats


def _safe_standardize(value: float, mean: float, std: float) -> float:
    """Standardize using precomputed mean/std and guard against zero std."""
    if std == 0:
        return value
    return (value - mean) / std


def prepare_model_input(
    model_features: List[str],
    template: Dict[str, float],
    stats: Dict[str, float],
    temperature: float,
    humidity: float,
    windspeed: float,
    season_value: int,
    weather_value: int,
) -> pd.DataFrame:
    """Create one inference row in the exact schema and order expected by the model."""
    # Start from learned feature averages so every expected column is present.
    row = {feature: float(template.get(feature, 0.0)) for feature in model_features}

    # Update time-dependent features with current context.
    now = datetime.now()
    row["hr"] = float(now.hour)
    row["day"] = float(now.day)
    row["month"] = float(now.month)
    row["mnth"] = float(now.month)
    row["weekday"] = float(now.weekday())
    row["workingday"] = float(1 if now.weekday() < 5 else 0)
    row["holiday"] = 0.0

    # Match preprocessing: these three columns were standardized during training.
    row["temp"] = _safe_standardize(temperature, stats["temp_mean"], stats["temp_std"])
    row["hum"] = _safe_standardize(humidity, stats["hum_mean"], stats["hum_std"])
    row["windspeed"] = _safe_standardize(windspeed, stats["windspeed_mean"], stats["windspeed_std"])

    # Keep apparent temperature aligned with selected temperature (atemp is unscaled in training data).
    row["atemp"] = temperature

    # One-hot encode season and weather exactly as training features expect.
    for idx in (1, 2, 3, 4):
        season_col = f"season_{idx}"
        weather_col = f"weathersit_{idx}"
        if season_col in row:
            row[season_col] = 1.0 if season_value == idx else 0.0
        if weather_col in row:
            row[weather_col] = 1.0 if weather_value == idx else 0.0

    # Build dataframe with strict feature order required by the model.
    return pd.DataFrame([[row[col] for col in model_features]], columns=model_features)


def main() -> None:
    st.set_page_config(page_title="Bike Rental Predictor", page_icon="🚲", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(56, 189, 248, 0.12), transparent 34%),
                radial-gradient(circle at 88% 10%, rgba(59, 130, 246, 0.14), transparent 30%),
                linear-gradient(180deg, #020617 0%, #0b1220 100%);
            color: #e2e8f0;
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070d19 0%, #111c31 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="stSidebar"] * {
            color: #dbe7ff;
        }
        [data-testid="stSidebar"] .stSlider > div[data-baseweb="slider"] div[role="slider"] {
            border: 2px solid #ffffff;
            background: #0ea5e9;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: #2563eb;
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.2px;
            height: 2.9rem;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #1d4ed8;
        }
        .main-card {
            background: #111827;
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.95rem;
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.35);
            font-weight: 500;
        }
        .hero {
            background: #0f172a;
            color: #f8fafc;
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.4);
        }
        .hero-kicker {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.88;
            margin-bottom: 0.25rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2rem;
            line-height: 1.2;
            font-weight: 750;
            margin: 0;
        }
        .hero-sub {
            margin-top: 0.45rem;
            color: rgba(241, 245, 249, 0.92);
        }
        .context-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.65rem;
            margin-bottom: 0.95rem;
        }
        .context-card {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 0.72rem 0.85rem;
        }
        .context-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }
        .context-value {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 700;
        }
        [data-testid="stMetric"] {
            background: #0f172a;
            border: 1px solid rgba(59, 130, 246, 0.38);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(2, 6, 23, 0.4);
        }
        [data-testid="stMetricLabel"] {
            color: #cbd5e1;
            font-weight: 600;
            font-size: 0.95rem;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 2.25rem;
            font-weight: 800;
            line-height: 1.1;
        }
        [data-testid="stExpander"] {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 12px;
        }
        [data-testid="stExpander"] summary {
            color: #dbe7ff;
        }
        [data-testid="stExpanderDetails"] {
            background: #0b1220;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model(MODEL_PATH)
        model_features = list(model.feature_names_in_)
        template, stats = load_feature_reference(DATA_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Startup error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected startup error: {exc}")
        st.stop()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Demand Intelligence</div>
            <h1 class="hero-title">Bike Rental Demand Dashboard</h1>
            <div class="hero-sub">Predict hourly bike rentals with a tuned XGBoost model in a single click.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input Controls")
        st.markdown("Tune weather conditions and context.")
        st.markdown("---")
        st.markdown("**Weather Signals**")

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
        humidity = st.slider("Humidity", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
        windspeed = st.slider("Windspeed", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

        season_map = {
            "Spring": 1,
            "Summer": 2,
            "Fall": 3,
            "Winter": 4,
        }
        weather_map = {
            "Clear / Few clouds": 1,
            "Mist + Cloudy": 2,
            "Light snow or rain": 3,
            "Heavy rain / severe": 4,
        }

        st.markdown("**Context Tags**")
        season_label = st.selectbox("Season", options=list(season_map.keys()), index=1)
        weather_label = st.selectbox("Weather", options=list(weather_map.keys()), index=0)

        st.markdown("---")
        predict_btn = st.button("Predict Rentals", type="primary", use_container_width=True)

    st.markdown(
        '<div class="main-card">Model-ready features are assembled in the exact training order before inference, ensuring reliable predictions.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="context-grid">
            <div class="context-card">
                <div class="context-label">Season</div>
                <div class="context-value">{season_label}</div>
            </div>
            <div class="context-card">
                <div class="context-label">Weather</div>
                <div class="context-value">{weather_label}</div>
            </div>
            <div class="context-card">
                <div class="context-label">Temperature</div>
                <div class="context-value">{temperature:.2f}</div>
            </div>
            <div class="context-card">
                <div class="context-label">Humidity</div>
                <div class="context-value">{humidity:.2f}</div>
            </div>
            <div class="context-card">
                <div class="context-label">Windspeed</div>
                <div class="context-value">{windspeed:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    input_df = prepare_model_input(
        model_features=model_features,
        template=template,
        stats=stats,
        temperature=temperature,
        humidity=humidity,
        windspeed=windspeed,
        season_value=season_map[season_label],
        weather_value=weather_map[weather_label],
    )

    if predict_btn:
        prediction = float(model.predict(input_df)[0])
        st.session_state["last_prediction"] = max(0.0, prediction)

    current_prediction = st.session_state.get("last_prediction")

    left, right = st.columns([1.2, 1])
    with left:
        metric_value = "Click Predict Rentals" if current_prediction is None else f"{current_prediction:,.0f} bikes"
        st.metric(label="Predicted Hourly Rentals", value=metric_value)
    with right:
        with st.expander("Model Input Preview", expanded=False):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
