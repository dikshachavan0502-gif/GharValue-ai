
"""GharValue AI — Modern Real Estate Price Prediction."""

from pathlib import Path
import html
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "gharvalue_linear_regression.joblib"
SUMMARY_FILE = ROOT / "models" / "project_summary.json"


st.set_page_config(
    page_title="GharValue AI | Property Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# DESIGN
# =========================================================

def apply_design():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --navy: #122b45;
            --teal: #087e8b;
            --mint: #12b8a6;
            --muted: #71869b;
            --border: #e2ebf2;
        }

        .stApp {
            background:
                radial-gradient(ellipse at 0% 0%,
                    rgba(18,184,166,.065), transparent 30%),
                #f5f8fc;
            color: var(--navy);
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(245,248,252,.88);
        }

        .block-container {
            max-width: 1320px;
            padding: 2rem 2.2rem 3rem;
        }

        h1, h2, h3, h4 {
            color: var(--navy) !important;
            font-family: 'Manrope', sans-serif !important;
            letter-spacing: -.035em;
        }

        p {
            line-height: 1.7;
        }

        /* Keep labels and values visible in light theme */

        .stApp label,
        .stApp label *,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] *,
        .stApp [data-baseweb="select"] *,
        .stApp [data-baseweb="input"] input {
            color: #263f56 !important;
            -webkit-text-fill-color: #263f56;
            opacity: 1;
        }

        /* HERO */

        .hero {
            position: relative;
            overflow: hidden;
            box-sizing: border-box;
            width: 100%;
            padding: 3rem 3.3rem;
            margin: 0 auto 2.5rem;
            border-radius: 26px;
            background:
                radial-gradient(circle at 92% 15%,
                    rgba(18,184,166,.32), transparent 27%),
                linear-gradient(120deg, #112b45, #174f64 65%, #087e8b);
            box-shadow: 0 18px 45px rgba(18,43,69,.12);
            animation: rise-in .55s ease-out both;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            right: -75px;
            bottom: -125px;
            border: 1px solid rgba(255,255,255,.23);
            border-radius: 50%;
            box-shadow:
                0 0 0 25px rgba(255,255,255,.045),
                0 0 0 50px rgba(255,255,255,.025);
            pointer-events: none;
        }

        .hero-tag {
            display: inline-block;
            padding: 8px 13px;
            border: 1px solid rgba(255,255,255,.2);
            border-radius: 30px;
            background: rgba(255,255,255,.09);
            color: #b8fff2 !important;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .12em;
        }

        .hero-title {
            margin: 1.2rem 0 .8rem;
            max-width: 800px;
            color: #fff !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: clamp(2.2rem, 4vw, 3.5rem);
            font-weight: 800;
            line-height: 1.14;
            letter-spacing: -.055em;
        }

        .hero-description {
            max-width: 710px;
            margin: 0 0 1.5rem;
            color: #d8e9f1 !important;
            font-size: 1rem;
        }

        .hero-chip {
            display: inline-block;
            margin: 0 .4rem .3rem 0;
            padding: 8px 12px;
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 9px;
            background: rgba(255,255,255,.09);
            color: #fff !important;
            font-size: .79rem;
        }

        /* SECTION HEADERS */

        .eyebrow {
            margin: 0 0 .35rem;
            color: var(--teal) !important;
            font-size: .73rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
        }

        .section-title {
            margin: 0 0 .35rem;
            color: var(--navy) !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -.045em;
        }

        .section-subtitle {
            margin: 0 0 1.25rem;
            color: var(--muted) !important;
            font-size: .92rem;
        }

        /* PREDICTION FORM PANEL */

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border) !important;
            border-radius: 20px !important;
            background: #fff;
            box-shadow: 0 7px 26px rgba(18,43,69,.045);
        }

        .form-heading {
            margin: 0 0 .3rem;
            color: var(--navy) !important;
            font-size: 1.08rem;
            font-weight: 800;
        }

        .form-caption {
            margin: 0 0 1rem;
            color: var(--muted) !important;
            font-size: .85rem;
        }

        .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div {
            min-height: 44px;
            border-radius: 10px !important;
            border-color: #dce6ed !important;
        }

        .stApp [data-baseweb="select"] > div:focus-within,
        .stApp [data-baseweb="input"] > div:focus-within {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 2px rgba(8,126,139,.12);
        }

        .stApp [data-testid="stNumberInput"] button {
            color: var(--teal) !important;
        }

        .stApp [data-testid="stFormSubmitButton"] button {
            min-height: 3.15rem;
            border: 0;
            border-radius: 11px;
            background: linear-gradient(110deg, #087e8b, #12a99b);
            box-shadow: 0 7px 18px rgba(8,126,139,.18);
            transition: transform .2s ease, box-shadow .2s ease;
        }

        .stApp [data-testid="stFormSubmitButton"] button,
        .stApp [data-testid="stFormSubmitButton"] button p {
            color: #fff !important;
            font-weight: 800 !important;
        }

        .stApp [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(8,126,139,.25);
        }

        /* PREDICTION RESULT */

        .result-card {
            padding: 1.7rem;
            margin: 1.2rem 0 0;
            border: 1px solid #c6eae4;
            border-radius: 19px;
            background: linear-gradient(135deg, #e7f8f4, #f5faff);
            box-shadow: 0 8px 24px rgba(18,43,69,.045);
            animation: rise-in .4s ease-out both;
        }

        .result-label {
            color: var(--teal) !important;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .12em;
        }

        .result-price {
            margin: .6rem 0;
            color: var(--navy) !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: -.045em;
            overflow-wrap: anywhere;
        }

        .result-detail {
            color: #526d83 !important;
            font-size: .88rem;
        }

        /* MODEL PERFORMANCE */

        .metric-card {
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            height: 168px;
            min-width: 0;
            padding: 1.2rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: #fff;
            box-shadow: 0 6px 22px rgba(18,43,69,.035);
            transition:
                transform .23s ease,
                box-shadow .23s ease,
                border-color .23s ease;
            animation: rise-in .5s ease-out both;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #9bdcd4;
            box-shadow: 0 14px 28px rgba(18,43,69,.09);
        }

        .metric-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .4rem;
            margin-bottom: auto;
        }

        .metric-label {
            color: #5d748b !important;
            font-size: .79rem;
            font-weight: 800;
            letter-spacing: .045em;
        }

        .metric-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            flex-shrink: 0;
            border-radius: 10px;
            background: #e6f7f4;
            color: #087e8b !important;
            font-size: .88rem;
            font-weight: 800;
        }

        .metric-value {
            display: block;
            margin-top: 1rem;
            color: #122b45 !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: clamp(1rem, 1.45vw, 1.45rem);
            font-weight: 800;
            line-height: 1.35;
            letter-spacing: -.04em;
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
        }

        .metric-description {
            margin-top: .3rem;
            color: #8194a8 !important;
            font-size: .7rem;
            line-height: 1.4;
        }

        /* FOOTER */

        .footer {
            margin-top: 2.5rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--border);
            color: #8194a8 !important;
            text-align: center;
            font-size: .78rem;
            line-height: 1.8;
        }

        @keyframes rise-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 1.2rem 1.1rem 2rem;
            }

            .hero {
                padding: 2.2rem 1.6rem;
                border-radius: 21px;
            }

            .metric-card {
                height: 150px;
                padding: .9rem;
            }

            .metric-value {
                font-size: clamp(.9rem, 2vw, 1.2rem);
            }

            .metric-description {
                font-size: .65rem;
            }
        }

        @media (max-width: 500px) {
            .hero-title {
                font-size: 2.15rem;
            }

            .metric-card {
                height: 142px;
                padding: .75rem;
            }

            .metric-value {
                font-size: .95rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation: none !important;
                transition: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="Loading the trained model...")
def load_assets():
    missing = [
        str(path.relative_to(ROOT))
        for path in (MODEL_FILE, SUMMARY_FILE)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing)
        )

    model = joblib.load(MODEL_FILE)
    summary = json.loads(
        SUMMARY_FILE.read_text(encoding="utf-8")
    )

    return model, summary


# =========================================================
# PREDICTION FUNCTIONS
# =========================================================

def format_inr(value):
    return f"₹{value:,.0f}"


def compact_inr(value):
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} crore"
    if value >= 100_000:
        return f"₹{value / 100_000:.2f} lakh"
    return format_inr(value)


def predict_price(model, summary, city, locality, area, bhk, bathrooms):
    if (
        city not in summary.get("cities", {})
        or locality not in summary["cities"].get(city, [])
    ):
        raise ValueError("Please select a supported city and locality.")

    if area <= 0 or bhk <= 0 or bathrooms <= 0:
        raise ValueError("All property values must be positive.")

    features = pd.DataFrame(
        [[f"{city} | {locality}", area, bhk, bathrooms]],
        columns=["location", "area_sqft", "bhk", "bathrooms"],
    )

    prediction = float(model.predict(features)[0])
    floor = float(summary.get("city_price_floors", {}).get(city, 1))

    return max(floor, prediction)


# =========================================================
# PERFORMANCE CARD
# =========================================================

def metric_card(label, value, icon, description):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">{html.escape(str(label))}</span>
                <span class="metric-icon">{html.escape(str(icon))}</span>
            </div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-description">
                {html.escape(str(description))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# APPLICATION
# =========================================================

def main():
    apply_design()

    try:
        model, summary = load_assets()
    except Exception as error:
        st.error(f"Unable to load the trained model: {error}")
        st.stop()

    # 1. HERO BANNER

    st.markdown(
        """
        <div class="hero">
            <span class="hero-tag">
                ✦ MACHINE LEARNING · REAL ESTATE INTELLIGENCE
            </span>
            <div class="hero-title">
                Find the value<br>behind every address.
            </div>
            <p class="hero-description">
                Estimate residential property prices with GharValue AI.
                Enter a location and property details to generate an
                estimate using your trained Linear Regression model.
            </p>
            <span class="hero-chip">⌂ Property prediction</span>
            <span class="hero-chip">◈ Data-driven insights</span>
            <span class="hero-chip">✓ Model evaluation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. PROPERTY ESTIMATOR — FIRST

    st.markdown(
        """
        <p class="eyebrow">PROPERTY ESTIMATOR</p>
        <div class="section-title">Estimate your property's value</div>
        <p class="section-subtitle">
            Configure the property details below to get your price estimate.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cities = list(summary.get("cities", {}).keys())

    if not cities:
        st.error("No supported cities were found in project_summary.json.")
        st.stop()

    with st.container(border=True):
        st.markdown(
            """
            <div class="form-heading">Property details</div>
            <p class="form-caption">
                Select your location and enter the home's specifications.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            # Aligned location row

            city_col, locality_col = st.columns([1, 1.5], gap="large")

            with city_col:
                city = st.selectbox("City", options=cities)

            localities = summary.get("cities", {}).get(city, [])

            if not localities:
                st.error("No localities are available for this city.")
                st.stop()

            with locality_col:
                locality = st.selectbox(
                    "Locality / area",
                    options=localities,
                )

            # Aligned property details row

            area_col, bhk_col, bathroom_col = st.columns(
                3, gap="large"
            )

            with area_col:
                area = st.number_input(
                    "Area (sq ft)",
                    min_value=1.0,
                    max_value=1_000_000.0,
                    value=1200.0,
                    step=50.0,
                )

            with bhk_col:
                bhk = st.number_input(
                    "Bedrooms (BHK)",
                    min_value=1.0,
                    max_value=30.0,
                    value=2.0,
                    step=1.0,
                )

            with bathroom_col:
                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1.0,
                    max_value=30.0,
                    value=2.0,
                    step=1.0,
                )

            submitted = st.form_submit_button(
                "✦   Predict property price",
                type="primary",
                use_container_width=True,
            )

    if submitted:
        try:
            value = predict_price(
                model, summary, city, locality, area, bhk, bathrooms
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        YOUR ESTIMATED PROPERTY VALUE
                    </div>
                    <div class="result-price">
                        {html.escape(format_inr(value))}
                    </div>
                    <div class="result-detail">
                        Approximately {html.escape(compact_inr(value))}
                    </div>
                    <p class="result-detail">
                        {html.escape(locality)}, {html.escape(city)}
                        &nbsp; · &nbsp; {area:,.0f} sq ft
                        &nbsp; · &nbsp; {bhk:g} BHK
                        &nbsp; · &nbsp; {bathrooms:g} bathroom(s)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This is a model-generated estimate, not a guaranteed "
                "market price or formal property valuation."
            )

        except (ValueError, TypeError) as error:
            st.error(str(error))
        except Exception:
            st.error(
                "Prediction failed. Please verify your model and input data."
            )

    # 3. MODEL PERFORMANCE — BELOW THE ESTIMATOR

    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class="eyebrow">MODEL PERFORMANCE</p>
        <div class="section-title">Evaluation results</div>
        <p class="section-subtitle">
            Actual hold-out test metrics from your trained model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    metrics = summary["metrics"]

    metric_columns = st.columns(4, gap="medium")

    cards = [
        (
            "MAE",
            format_inr(metrics["mae"]),
            "₹",
            "Mean Absolute Error",
        ),
        (
            "RMSE",
            format_inr(metrics["rmse"]),
            "↗",
            "Root Mean Squared Error",
        ),
        (
            "MSE",
            f'₹{metrics["mse"]:,.0f}',
            "Σ",
            "Mean Squared Error",
        ),
        (
            "R² SCORE",
            f'{metrics["r2"]:.3f}',
            "R²",
            "Coefficient of determination",
        ),
    ]

    for column, card in zip(metric_columns, cards):
        with column:
            metric_card(*card)

    # 4. COMPACT FOOTER — NO EXTRA CONTENT SECTIONS

    st.markdown(
        """
        <div class="footer">
            <strong style="color:#087e8b;">GharValue AI</strong>
            &nbsp;·&nbsp; Real Estate Price Prediction
            <br>
            Machine Learning Project
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
