
"""GharValue AI — Indian Real Estate Price Prediction Platform."""

from pathlib import Path
import html
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "gharvalue_linear_regression.joblib"
SUMMARY_FILE = ROOT / "models" / "project_summary.json"


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="GharValue AI | Property Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# CUSTOM DESIGN SYSTEM
# ---------------------------------------------------------

def apply_design():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --navy: #102a43;
            --navy-light: #243b53;
            --teal: #087e8b;
            --teal-bright: #12b8a6;
            --muted: #627d98;
            --border: #e4edf2;
            --surface: #ffffff;
            --background: #f5f8fb;
        }

        /* APP BACKGROUND */

        .stApp {
            background:
                radial-gradient(ellipse at 5% 0%,
                    rgba(18,184,166,0.07), transparent 32%),
                #f5f8fb !important;
            color: var(--navy) !important;
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(245,248,251,0.92);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        /* GLOBAL TEXT VISIBILITY */

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4 {
            color: var(--navy) !important;
            font-family: 'Manrope', sans-serif !important;
            letter-spacing: -0.035em;
        }

        .stApp p,
        .stApp li {
            color: var(--navy-light);
        }

        .stApp label,
        .stApp label *,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] *,
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] * {
            color: var(--navy-light) !important;
            -webkit-text-fill-color: var(--navy-light) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* HERO */

        .hero {
            position: relative;
            overflow: hidden;
            padding: 3rem 3.2rem;
            border-radius: 28px;
            margin-bottom: 1.7rem;
            background:
                radial-gradient(circle at 88% 12%,
                    rgba(18,184,166,0.28), transparent 27%),
                radial-gradient(circle at 72% 110%,
                    rgba(66,153,225,0.24), transparent 35%),
                linear-gradient(125deg, #102a43 0%,
                    #164e63 57%, #087e8b 100%);
            box-shadow: 0 20px 50px rgba(16,42,67,0.15);
            animation: hero-enter 0.65s ease-out both;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 250px;
            height: 250px;
            right: -85px;
            bottom: -130px;
            border: 1px solid rgba(255,255,255,0.20);
            border-radius: 50%;
            box-shadow:
                0 0 0 28px rgba(255,255,255,0.035),
                0 0 0 56px rgba(255,255,255,0.025);
            pointer-events: none;
        }

        .hero-kicker {
            display: inline-block;
            padding: 7px 12px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.09);
            border-radius: 30px;
            color: #b7fff3 !important;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .hero-title {
            color: #ffffff !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: clamp(2.2rem, 4vw, 3.65rem);
            line-height: 1.12;
            letter-spacing: -0.055em;
            font-weight: 800;
            margin: 1.15rem 0 0.8rem;
        }

        .hero-description {
            color: #d9edf2 !important;
            font-size: 1.02rem;
            line-height: 1.8;
            max-width: 680px;
            margin-bottom: 1.4rem;
        }

        .hero-pill {
            display: inline-block;
            margin: 0.25rem 0.4rem 0 0;
            padding: 7px 12px;
            border-radius: 9px;
            background: rgba(255,255,255,0.10);
            color: #ffffff !important;
            font-size: 0.78rem;
            border: 1px solid rgba(255,255,255,0.12);
        }

        /* SECTION HEADERS */

        .section-kicker {
            color: var(--teal) !important;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 0 0 0.4rem;
        }

        .section-description {
            color: var(--muted) !important;
            font-size: 0.92rem;
            line-height: 1.7;
        }

        /* METRIC CARDS */

        .metric-card {
            box-sizing: border-box;
            height: 100%;
            min-height: 155px;
            padding: 1.25rem 1.15rem;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 19px;
            box-shadow: 0 5px 18px rgba(16,42,67,0.035);
            transition:
                transform 0.25s ease,
                box-shadow 0.25s ease,
                border-color 0.25s ease;
            animation: card-enter 0.55s ease-out both;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #9adbd4;
            box-shadow: 0 15px 32px rgba(16,42,67,0.10);
        }

        .metric-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 1.2rem;
        }

        .metric-label {
            color: #526a80 !important;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .metric-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 35px;
            height: 35px;
            flex-shrink: 0;
            border-radius: 11px;
            background: #e8f8f5;
            color: #087e8b !important;
            font-size: 1rem;
            font-weight: 800;
        }

        .metric-value {
            color: #102a43 !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: clamp(1.2rem, 1.8vw, 1.8rem);
            font-weight: 800;
            line-height: 1.3;
            letter-spacing: -0.045em;
            overflow-wrap: anywhere;
        }

        .metric-foot {
            margin-top: 0.45rem;
            color: #8195a8 !important;
            font-size: 0.72rem;
            line-height: 1.5;
        }

        /* PANEL CARDS */

        .panel-heading {
            color: var(--navy) !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: 1.22rem;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin: 0 0 0.4rem;
        }

        .panel-description {
            color: var(--muted) !important;
            font-size: 0.87rem;
            line-height: 1.7;
            margin-bottom: 0.8rem;
        }

        .panel-badge {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 8px;
            background: #e8f8f5;
            color: #087e8b !important;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        /* FORM WIDGETS */

        .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div {
            border-radius: 11px !important;
        }

        .stApp [data-baseweb="select"] > div:focus-within,
        .stApp [data-baseweb="input"] > div:focus-within {
            border-color: #087e8b !important;
            box-shadow: 0 0 0 2px rgba(8,126,139,0.13) !important;
        }

        .stApp [data-baseweb="select"] *,
        .stApp [data-baseweb="input"] input {
            opacity: 1 !important;
        }

        .stApp [data-baseweb="select"] input,
        .stApp [data-baseweb="input"] input {
            color: #102a43 !important;
            -webkit-text-fill-color: #102a43 !important;
        }

        .stApp [data-testid="stNumberInput"] button {
            color: #087e8b !important;
        }

        /* PREDICTION BUTTON */

        .stApp [data-testid="stFormSubmitButton"] button {
            min-height: 3.2rem;
            border: 0 !important;
            border-radius: 12px !important;
            background: linear-gradient(110deg, #087e8b, #12a99b) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            box-shadow: 0 7px 18px rgba(8,126,139,0.18);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stApp [data-testid="stFormSubmitButton"] button p {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        .stApp [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(8,126,139,0.27);
        }

        /* PREDICTION RESULT */

        .result-card {
            padding: 1.6rem;
            margin-top: 1rem;
            background: linear-gradient(140deg, #e8f8f5, #f6fbff);
            border: 1px solid #c6ebe5;
            border-radius: 18px;
            animation: card-enter 0.45s ease-out both;
        }

        .result-kicker {
            color: #087e8b !important;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .result-price {
            color: #102a43 !important;
            font-family: 'Manrope', sans-serif !important;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.25;
            font-weight: 800;
            letter-spacing: -0.05em;
            overflow-wrap: anywhere;
            margin: 0.65rem 0;
        }

        .result-subtitle {
            color: #526a80 !important;
            font-size: 0.95rem;
        }

        .result-disclaimer {
            color: #627d98 !important;
            font-size: 0.78rem;
            line-height: 1.7;
            margin-top: 1rem;
        }

        /* EXPANDERS */

        .stApp [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: #ffffff;
        }

        .stApp [data-testid="stExpander"] summary {
            color: var(--navy) !important;
        }

        /* DIVIDERS */

        .stApp hr {
            border-color: #e0eaf0 !important;
        }

        /* FOOTER */

        .app-footer {
            text-align: center;
            padding: 1.6rem 0 0.5rem;
            color: #8195a8 !important;
            font-size: 0.78rem;
            line-height: 1.8;
        }

        /* ANIMATIONS */

        @keyframes hero-enter {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes card-enter {
            from {
                opacity: 0;
                transform: translateY(9px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* ACCESSIBILITY: respect reduced motion */

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* MOBILE RESPONSIVENESS */

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 2rem 1.4rem;
                border-radius: 20px;
            }

            .hero-title {
                font-size: 2.25rem;
            }

            .metric-card {
                min-height: 135px;
                padding: 1rem 0.85rem;
            }

            .metric-value {
                font-size: 1.15rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading GharValue AI model...")
def load_assets():
    missing = [
        str(path.relative_to(ROOT))
        for path in (MODEL_FILE, SUMMARY_FILE)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing deployment files: "
            + ", ".join(missing)
            + ". Check your GitHub models/ folder."
        )

    model = joblib.load(MODEL_FILE)
    summary = json.loads(
        SUMMARY_FILE.read_text(encoding="utf-8")
    )

    return model, summary


# ---------------------------------------------------------
# PRICE UTILITIES
# ---------------------------------------------------------

def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def compact_inr(value: float) -> str:
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} crore"
    if value >= 100_000:
        return f"₹{value / 100_000:.2f} lakh"
    return format_inr(value)


def predict_price(
    model,
    summary,
    city: str,
    locality: str,
    area: float,
    bhk: float,
    bathrooms: float,
) -> float:
    """Generate a prediction using the existing trained pipeline."""

    if (
        city not in summary.get("cities", {})
        or locality not in summary["cities"].get(city, [])
    ):
        raise ValueError("Please select a supported city and locality.")

    if area <= 0 or bhk <= 0 or bathrooms <= 0:
        raise ValueError(
            "Area, bedrooms, and bathrooms must be positive."
        )

    features = pd.DataFrame(
        [[f"{city} | {locality}", area, bhk, bathrooms]],
        columns=[
            "location",
            "area_sqft",
            "bhk",
            "bathrooms",
        ],
    )

    prediction = float(model.predict(features)[0])

    floor = float(
        summary.get("city_price_floors", {}).get(city, 1)
    )

    return max(floor, prediction)


# ---------------------------------------------------------
# REUSABLE METRIC CARD
# ---------------------------------------------------------

def metric_card(label, value, icon, description):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_icon = html.escape(str(icon))
    safe_description = html.escape(str(description))

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">{safe_label}</span>
                <span class="metric-icon">{safe_icon}</span>
            </div>
            <div class="metric-value">{safe_value}</div>
            <div class="metric-foot">{safe_description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

def main():
    apply_design()

    try:
        model, summary = load_assets()
    except Exception as error:
        st.error(f"Unable to load the trained model: {error}")
        st.stop()

    # HERO BANNER

    st.markdown(
        """
        <section class="hero">
            <span class="hero-kicker">
                ✦ Machine Learning · Real Estate Intelligence
            </span>
            <div class="hero-title">
                Find the value<br>behind every address.
            </div>
            <p class="hero-description">
                Estimate residential property prices with GharValue AI.
                Explore property locations, enter home specifications,
                and generate an estimate from a trained Linear Regression
                model.
            </p>
            <span class="hero-pill">⌂ Property prediction</span>
            <span class="hero-pill">◈ Data-driven insights</span>
            <span class="hero-pill">✓ Model evaluation</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # MODEL SUMMARY

    st.markdown(
        '<p class="section-kicker">MODEL PERFORMANCE</p>'
        '<div class="panel-heading">At a glance</div>'
        '<p class="section-description">'
        'Evaluation results from the saved hold-out test metrics.'
        '</p>',
        unsafe_allow_html=True,
    )

    metrics = summary["metrics"]

    metric_cols = st.columns(4, gap="medium")

    cards = [
        (
            "MAE",
            format_inr(metrics["mae"]),
            "₹",
            "Mean Absolute Error · Lower is better",
        ),
        (
            "RMSE",
            format_inr(metrics["rmse"]),
            "↗",
            "Root Mean Squared Error · Lower is better",
        ),
        (
            "MSE",
            format_inr(metrics["mse"]),
            "Σ",
            "Mean Squared Error · Lower is better",
        ),
        (
            "R² SCORE",
            f'{metrics["r2"]:.3f}',
            "R²",
            "Coefficient of determination",
        ),
    ]

    for column, card in zip(metric_cols, cards):
        with column:
            metric_card(*card)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    # PREDICTION SECTION

    prediction_col, info_col = st.columns(
        [1.35, 0.85],
        gap="large",
    )

    with prediction_col:
        st.markdown(
            """
            <p class="section-kicker">PROPERTY ESTIMATOR</p>
            <div class="panel-heading">
                Configure your property
            </div>
            <p class="panel-description">
                Enter the property details below to estimate its price.
                Choose a locality that is available for your selected city.
            </p>
            <span class="panel-badge">● Prediction workspace</span>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 0.8rem'></div>",
                    unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                '<p class="panel-heading">01 · Location details</p>',
                unsafe_allow_html=True,
            )

            cities = list(summary["cities"].keys())

            if not cities:
                st.error("No supported cities were found in the model summary.")
                st.stop()

            with st.form("prediction_form"):
                city = st.selectbox(
                    "City",
                    options=cities,
                    help="Choose a city supported by the trained model.",
                )

                localities = summary["cities"][city]

                if not localities:
                    st.error("No localities are available for this city.")
                    st.stop()

                locality = st.selectbox(
                    "Locality / area",
                    options=localities,
                    help="Select a supported locality.",
                )

                st.markdown(
                    '<p class="panel-heading">02 · Property details</p>',
                    unsafe_allow_html=True,
                )

                area_col, bedroom_col = st.columns(2)

                with area_col:
                    area = st.number_input(
                        "Area (sq ft)",
                        min_value=1.0,
                        max_value=1_000_000.0,
                        value=1200.0,
                        step=50.0,
                        help="Enter the property area in square feet.",
                    )

                with bedroom_col:
                    bhk = st.number_input(
                        "Bedrooms (BHK)",
                        min_value=1.0,
                        max_value=30.0,
                        value=2.0,
                        step=1.0,
                    )

                bathroom_col, _ = st.columns(2)

                with bathroom_col:
                    bathrooms = st.number_input(
                        "Bathrooms",
                        min_value=1.0,
                        max_value=30.0,
                        value=2.0,
                        step=1.0,
                    )

                submitted = st.form_submit_button(
                    "✦  Predict property price",
                    type="primary",
                    use_container_width=True,
                )

        if submitted:
            try:
                value = predict_price(
                    model,
                    summary,
                    city,
                    locality,
                    area,
                    bhk,
                    bathrooms,
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-kicker">
                            YOUR ESTIMATED PROPERTY VALUE
                        </div>
                        <div class="result-price">
                            {html.escape(format_inr(value))}
                        </div>
                        <div class="result-subtitle">
                            Approximately
                            {html.escape(compact_inr(value))}
                        </div>
                        <p class="result-disclaimer">
                            {html.escape(locality)}, {html.escape(city)}
                            · {area:,.0f} sq ft · {bhk:g} BHK
                            · {bathrooms:g} bathroom(s)
                            <br><br>
                            This is a model-generated educational estimate,
                            not a guaranteed sale price or formal valuation.
                            Actual prices can vary with market conditions,
                            property condition, and other factors.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except (ValueError, TypeError) as error:
                st.error(str(error))
            except Exception:
                st.error(
                    "The prediction could not be completed. "
                    "Please check the inputs and try again."
                )

    # SIDE INFORMATION PANEL

    with info_col:
        st.markdown(
            """
            <p class="section-kicker">HOW IT WORKS</p>
            <div class="panel-heading">From details to estimate</div>
            <p class="panel-description">
                A simple workflow powered by your trained ML model.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                """
                <div style="padding: 0.4rem 0;">
                    <div style="
                        width: 38px; height: 38px;
                        display:flex; align-items:center;
                        justify-content:center;
                        border-radius:12px;
                        background:#e8f8f5;
                        color:#087e8b;
                        font-weight:800;">
                        01
                    </div>
                    <h4 style="margin: 0.8rem 0 0.3rem;">
                        Select a location
                    </h4>
                    <p style="color:#627d98; font-size:0.88rem;
                        line-height:1.7;">
                        Choose a supported city and locality from
                        the model's available location data.
                    </p>
                    <hr style="border:0;border-top:1px solid #e4edf2;">
                    <div style="
                        width: 38px; height: 38px;
                        display:flex; align-items:center;
                        justify-content:center;
                        border-radius:12px;
                        background:#e8f8f5;
                        color:#087e8b;
                        font-weight:800;">
                        02
                    </div>
                    <h4 style="margin: 0.8rem 0 0.3rem;">
                        Enter property details
                    </h4>
                    <p style="color:#627d98; font-size:0.88rem;
                        line-height:1.7;">
                        Provide the area, bedroom count, and
                        bathroom count.
                    </p>
                    <hr style="border:0;border-top:1px solid #e4edf2;">
                    <div style="
                        width: 38px; height: 38px;
                        display:flex; align-items:center;
                        justify-content:center;
                        border-radius:12px;
                        background:#e8f8f5;
                        color:#087e8b;
                        font-weight:800;">
                        03
                    </div>
                    <h4 style="margin: 0.8rem 0 0.3rem;">
                        Generate your estimate
                    </h4>
                    <p style="color:#627d98; font-size:0.88rem;
                        line-height:1.7;">
                        The saved Linear Regression pipeline
                        produces a price estimate from your inputs.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1rem'></div>",
                    unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                """
                <p class="section-kicker">IMPORTANT NOTE</p>
                <div class="panel-heading">Use results responsibly</div>
                <p style="color:#627d98;font-size:0.87rem;line-height:1.8;">
                    Machine learning estimates are not professional
                    property valuations. Model performance depends on
                    the quality, coverage, and representativeness of
                    the training data.
                </p>
                """,
                unsafe_allow_html=True,
            )

    # DATASET AND MODEL INFORMATION

    st.markdown("<div style='height: 1.8rem'></div>",
                unsafe_allow_html=True)

    st.markdown(
        """
        <p class="section-kicker">PROJECT TRANSPARENCY</p>
        <div class="panel-heading">Dataset & model information</div>
        <p class="section-description">
            Review the source dataset and details of the trained model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View dataset and training information"):
        dataset = summary.get("dataset", {})
        cleaning = summary.get("cleaning", {})

        st.markdown("**Dataset source**")

        if dataset.get("source"):
            source = html.escape(str(dataset["source"]), quote=True)
            name = html.escape(str(dataset.get("name", "Dataset")))
            st.markdown(f"[{name}]({source})")
        else:
            st.write(dataset.get("name", "Not specified"))

        st.markdown("**Training data**")
        rows = cleaning.get("rows_after_cleaning")

        if rows is not None:
            st.write(f"Cleaned records: {rows:,}")
        else:
            st.write("Cleaned record count is not available.")

        st.markdown("**Model type**")
        st.write("Linear Regression")

        st.markdown("**Deployment model**")
        st.code("models/gharvalue_linear_regression.joblib")

        st.caption(
            "The deployed app loads the saved model and does not retrain "
            "it on each startup."
        )

    # FOOTER

    st.markdown(
        """
        <div class="app-footer">
            <strong style="color:#087e8b;">GharValue AI</strong>
            · Machine Learning for Real Estate
            <br>
            Academic project demonstration · Estimates are indicative only
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
