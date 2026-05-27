"""
app.py
------
Streamlit app for preliminary prediction of HFMN microneedle properties.

Usage:
    streamlit run app.py

Loads trained models from the models/ folder and provides an interactive
interface to enter formulation, process, and geometry parameters.
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGET_LABELS = {
    "swelling_pct_mean": "Swelling (%)",
    "gel_fraction_pct_mean": "Gel Fraction (%)",
    "Fmax_N_per_cm2_mean": "Fmax (N/cm²)",
}

TARGET_UNITS = {
    "swelling_pct_mean": "%",
    "gel_fraction_pct_mean": "%",
    "Fmax_N_per_cm2_mean": "N/cm²",
}

# Default values for inputs (medians from the dataset where possible)
DEFAULT_VALUES = {
    "PVA_wt_pct": 15.0,
    "PVP_wt_pct": 5.0,
    "citric_acid_wt_pct": 1.5,
    "PMVE_MA_or_Gantrez_wt_pct": 15.0,
    "PEG10000_wt_pct": 7.5,
    "PEG400_wt_pct": 5.0,
    "glycerol_wt_pct": 5.0,
    "water_wt_pct": 82.0,
    "crosslink_temp_C": 130.0,
    "crosslink_time_min": 40.0,
    "needle_number_total": 196.0,
    "needle_height_um": 600.0,
    "needle_base_width_um": 300.0,
    "interspacing_um": 300.0,
    "film_thickness_mm": 0.5,
}

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

@st.cache_resource
def load_all_models():
    """Load all available trained models from the models/ folder."""
    models = {}
    if not os.path.isdir(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("model_") and fname.endswith(".joblib"):
            target = fname[len("model_"):-len(".joblib")]
            path = os.path.join(MODEL_DIR, fname)
            bundle = joblib.load(path)
            models[target] = bundle
    return models


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HFMN Predictor", page_icon=":test_tube:", layout="wide")

st.title("Hydrogel-Forming Microneedle (HFMN) Property Predictor")
st.caption("Early-stage screening tool based on literature-extracted data")

st.warning(
    "**Preliminary model — for early-stage screening only.** "
    "This model was trained on a small literature-extracted dataset (~25 data points) "
    "and is not suitable for final experimental validation. "
    "Use predictions as rough guidance for formulation exploration, "
    "not as a replacement for experimental measurement."
)

# Load models
models = load_all_models()

if not models:
    st.error(
        "No trained models found in the `models/` folder. "
        "Run `python train_model.py` first to train the models."
    )
    st.stop()

available_targets = sorted(models.keys())
st.info(f"Loaded models for: **{', '.join(TARGET_LABELS.get(t, t) for t in available_targets)}**")

# ── Sidebar: input parameters ──
st.sidebar.header("Input Parameters")

# Determine which features the first model expects
any_bundle = next(iter(models.values()))
expected_features = any_bundle.get("features", list(DEFAULT_VALUES.keys()))

st.sidebar.markdown("### Formulation")
form_values = {}
for feat in expected_features:
    if feat in ("needle_number_total", "needle_height_um", "needle_base_width_um",
                 "interspacing_um", "array_area_cm2", "film_thickness_mm"):
        continue  # geometry params go in a separate section
    default = float(DEFAULT_VALUES.get(feat, 0.0))
    step = 0.5 if default < 20 else 5.0
    form_values[feat] = st.sidebar.number_input(
        f"{feat}", min_value=0.0, value=default, step=step, format="%.1f"
    )

st.sidebar.markdown("### Process")
for feat in ["crosslink_temp_C", "crosslink_time_min"]:
    if feat not in expected_features:
        continue
    default = float(DEFAULT_VALUES.get(feat, 0.0))
    step = 5.0 if "temp" in feat else 10.0
    form_values[feat] = st.sidebar.number_input(
        f"{feat}", min_value=0.0, value=default, step=step, format="%.1f"
    )

st.sidebar.markdown("### Geometry")
for feat in ["needle_number_total", "needle_height_um", "needle_base_width_um",
             "interspacing_um", "film_thickness_mm"]:
    if feat not in expected_features:
        continue
    default = float(DEFAULT_VALUES.get(feat, 0.0))
    step = 1.0 if default < 10 else 50.0
    form_values[feat] = st.sidebar.number_input(
        f"{feat}", min_value=0.0, value=default, step=step, format="%.1f"
    )

# ── Predict ──
st.markdown("---")
st.subheader("Predictions")

if st.button("Run Prediction", type="primary"):
    cols = st.columns(len(available_targets) if len(available_targets) <= 3 else 3)
    for i, target in enumerate(available_targets):
        bundle = models[target]
        model = bundle["model"]
        imputer = bundle["imputer"]
        features = bundle["features"]

        # Build input vector (DataFrame to preserve feature names)
        X_input = pd.DataFrame([[form_values.get(f, 0.0) for f in features]], columns=features)
        X_input = imputer.transform(X_input)

        prediction = model.predict(X_input)[0]

        with cols[i % 3]:
            label = TARGET_LABELS.get(target, target)
            unit = TARGET_UNITS.get(target, "")
            st.metric(
                label=label,
                value=f"{prediction:.2f} {unit}",
            )

    st.caption(
        "These predictions are generated by a Random Forest model trained on "
        "~25 literature-extracted data points. Confidence intervals are not "
        "provided — treat results as rough approximations only."
    )

st.markdown("---")
st.markdown(
    "**Note:** The models were trained on a narrow formulation space. "
    "Input values far outside the ranges seen in the training data will "
    "produce unreliable extrapolations."
)
