from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/xgb_bike_model.joblib")
MODEL_CACHE_DIR = Path(".cache/models")
DEFAULT_MODEL_URL = "https://huggingface.co/dinesh-moorthy/bike-rental-model/resolve/main/xgb_bike_model.joblib"

# These values are derived from the training data and allow inference without loading data/hour.csv.
FEATURE_TEMPLATE_DEFAULTS: Dict[str, float] = {
    "yr": 0.5025605615973301,
    "mnth": 6.537775476149376,
    "hr": 11.546751826917545,
    "holiday": 0.028770355026181024,
    "weekday": 3.003682605443351,
    "workingday": 0.6827205247712756,
    "temp": 0.0,
    "atemp": 0.4757751021347604,
    "hum": 0.0,
    "windspeed": 0.0,
    "casual": 35.67621842453536,
    "registered": 153.78686920996606,
    "day": 15.683411013291904,
    "month": 6.537775476149376,
    "season_1": 0.2440876920421198,
    "season_2": 0.25369699062086426,
    "season_3": 0.25870303239541975,
    "season_4": 0.24351228494159619,
    "weathersit_1": 0.656712123827608,
    "weathersit_2": 0.26146498647793315,
    "weathersit_3": 0.08165026756430174,
    "weathersit_4": 0.00017262213015708613,
}

SCALER_STATS_DEFAULTS: Dict[str, float] = {
    "temp_mean": 0.4969871684216583,
    "temp_std": 0.19255058126205624,
    "hum_mean": 0.6272288394038783,
    "hum_std": 0.1929242833232444,
    "windspeed_mean": 0.1900976063064618,
    "windspeed_std": 0.12233670875034648,
}


def normalize_model_url(model_url: str, expected_filename: str) -> str:
    """Convert Hugging Face tree URL to a direct resolve URL when needed."""
    normalized = model_url.strip()

    if "/tree/" in normalized:
        normalized = normalized.replace("/tree/", "/resolve/")

    if normalized.endswith("/main") or normalized.endswith("/main/"):
        normalized = normalized.rstrip("/") + f"/{expected_filename}"

    return normalized


@st.cache_resource
def load_model(local_model_path: Path, model_url: str, cache_dir: Path):
    """Load and cache model, preferring local file and falling back to Hugging Face download."""
    if local_model_path.exists():
        return joblib.load(local_model_path), str(local_model_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model_path = cache_dir / local_model_path.name

    if not cached_model_path.exists():
        try:
            urlretrieve(model_url, cached_model_path)
        except URLError as exc:
            raise FileNotFoundError(
                "Model was not found locally and could not be downloaded from Hugging Face. "
                "Set MODEL_URL to a direct .joblib file URL if needed."
            ) from exc

    return joblib.load(cached_model_path), str(cached_model_path)


@st.cache_data
def load_feature_reference() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load static feature template and scaler stats for deployment-safe inference."""
    return FEATURE_TEMPLATE_DEFAULTS.copy(), SCALER_STATS_DEFAULTS.copy()


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

    model_url = normalize_model_url(os.getenv("MODEL_URL", DEFAULT_MODEL_URL), MODEL_PATH.name)

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
        model, loaded_model_path = load_model(MODEL_PATH, model_url, MODEL_CACHE_DIR)
        model_features = list(model.feature_names_in_)
        template, stats = load_feature_reference()
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
        st.caption(f"Model source: {loaded_model_path}")

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
