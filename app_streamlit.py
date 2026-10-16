"""Streamlit deployment entry point for GharValue AI.

This app only loads the already-trained model.  Use train_model.py separately
when the training data or model needs to be refreshed.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "gharvalue_linear_regression.joblib"
SUMMARY_FILE = ROOT / "models" / "project_summary.json"


@st.cache_resource(show_spinner="Loading the trained price model...")
def load_assets():
    """Load the deployment artifacts once per running Streamlit process."""
    missing = [str(path.relative_to(ROOT)) for path in (MODEL_FILE, SUMMARY_FILE) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required deployment artifact(s): " + ", ".join(missing) + ". "
            "Commit the existing models/ directory to the deployment repository."
        )
    return joblib.load(MODEL_FILE), json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def compact_inr(value: float) -> str:
    if value >= 10_000_000:
        return f"≈ ₹{value / 10_000_000:.2f} crore"
    return f"≈ ₹{value / 100_000:.2f} lakh"


def predict_price(model, summary, city: str, locality: str, area: float, bhk: float, bathrooms: float) -> float:
    """Run the same Linear Regression pipeline used by the Flask endpoint."""
    if city not in summary.get("cities", {}) or locality not in summary["cities"].get(city, []):
        raise ValueError("Choose a supported city and locality.")
    if area <= 0 or bhk <= 0 or bathrooms <= 0:
        raise ValueError("Area, BHK, and bathrooms must be positive values.")

    features = pd.DataFrame(
        [[f"{city} | {locality}", area, bhk, bathrooms]],
        columns=["location", "area_sqft", "bhk", "bathrooms"],
    )
    prediction = float(model.predict(features)[0])
    # Preserve the existing Flask app's protection against negative extrapolation.
    return max(float(summary.get("city_price_floors", {}).get(city, 1)), prediction)


def apply_design() -> None:
    """Reuse the existing navy/teal visual direction in Streamlit's layout."""
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; color: #243b53; }
        h1, h2, h3 { color: #102a43; }
        .hero { padding: 2.8rem 0 2rem; background: radial-gradient(circle at 80% 30%, #d5f1ef, transparent 35%); }
        .eyebrow { color: #087e8b; font-size: .75rem; font-weight: 700; letter-spacing: .13em; }
        .estimate { background: #f0fbfa; border: 1px solid #d5f1ef; border-radius: 14px; padding: 1.5rem; }
        .muted { color: #627d98; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="GharValue AI | House Price Prediction",
        page_icon="🏠",
        layout="wide",
    )
    apply_design()
    try:
        model, summary = load_assets()
    except Exception as error:
        st.error(f"Unable to load the trained model: {error}")
        st.stop()

    st.markdown('<div class="hero"><p class="eyebrow">INDIAN PROPERTY ESTIMATOR</p>', unsafe_allow_html=True)
    st.title("GharValue AI")
    st.subheader("Predict Your House Price")
    st.markdown("Estimate property prices in Bengaluru, Mumbai, and Pune using the saved Linear Regression model.")
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown('<p class="eyebrow">PRICE PREDICTION</p>', unsafe_allow_html=True)
        st.header("Tell us about the property")
        st.markdown('<p class="muted">Select an available locality for the chosen city, then enter the property details.</p>', unsafe_allow_html=True)
    with right:
        with st.form("prediction_form"):
            city = st.selectbox("City", options=list(summary["cities"].keys()))
            locality = st.selectbox("Locality / area", options=summary["cities"][city])
            first, second = st.columns(2)
            with first:
                area = st.number_input("Area (sq ft)", min_value=1.0, value=1200.0, step=50.0)
                bhk = st.number_input("Bedrooms (BHK)", min_value=1.0, value=2.0, step=1.0)
            with second:
                bathrooms = st.number_input("Bathrooms", min_value=1.0, value=2.0, step=1.0)
            submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

        if submitted:
            try:
                value = predict_price(model, summary, city, locality, area, bhk, bathrooms)
                st.markdown('<div class="estimate"><p class="eyebrow">YOUR EDUCATIONAL ESTIMATE</p>', unsafe_allow_html=True)
                st.markdown(f"## {format_inr(value)}")
                st.write(compact_inr(value))
                st.caption(f"{locality}, {city} · {area:,.0f} sq ft · {bhk:g} BHK · {bathrooms:g} bathroom(s)")
                st.caption("Model: Linear Regression · This is an educational estimate, not a guaranteed market price.")
                st.markdown("</div>", unsafe_allow_html=True)
            except (ValueError, TypeError) as error:
                st.error(str(error))
            except Exception:
                st.error("The prediction service could not process this request. Please try again.")

    st.divider()
    st.markdown('<p class="eyebrow">MODEL EVALUATION</p>', unsafe_allow_html=True)
    st.header("Actual hold-out test results")
    metrics = summary["metrics"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", format_inr(metrics["mae"]))
    metric_columns[1].metric("RMSE", format_inr(metrics["rmse"]))
    metric_columns[2].metric("MSE", format_inr(metrics["mse"]))
    metric_columns[3].metric("R² score", f"{metrics['r2']:.3f}")

    with st.expander("Dataset and model details"):
        dataset = summary["dataset"]
        st.write(f"**Dataset:** [{dataset['name']}]({dataset['source']})")
        st.write(f"**Cleaned records:** {summary['cleaning']['rows_after_cleaning']:,}")
        st.write("The deployed app uses the committed model artifact and does not train or download data on startup.")


if __name__ == "__main__":
    main()
