# HFMN Machine Learning — Preliminary Property Predictor

A simple, reproducible machine learning pipeline for predicting hydrogel-forming
microneedle (HFMN) properties from formulation, process, and geometry variables.

**Important:** This is a preliminary research workflow. The dataset contains
~25 literature-extracted data points. Models are suitable for early-stage
screening and formulation exploration only — not for final experimental
validation.

## What this project does

- Reads literature-extracted HFMN data from an Excel sheet (`Extracted_Data`).
- Trains separate regression models for up to three target properties:
  - **Swelling (%)** — equilibrium swelling ratio of the hydrogel.
  - **Gel Fraction (%)** — insoluble gel content after crosslinking.
  - **Fmax (N/cm²)** — maximum mechanical failure force per unit area.
- Compares Linear Regression and Random Forest Regression using
  Leave-One-Out cross-validation.
- Saves the best models (Random Forest) and generates diagnostic plots.
- Provides a Streamlit web app for interactive prediction.

## Project structure

```
├── requirements.txt              # Python dependencies
├── train_model.py                # Model training script
├── app.py                        # Streamlit interactive app
├── README.md                     # This file
├── models/                       # Saved trained models (*.joblib)
├── outputs/                      # Model performance & plots
│   ├── model_performance.xlsx
│   ├── feature_importance_each_target.xlsx
│   ├── actual_vs_predicted_*.png
│   └── feature_importance_*.png
└── Ryan_Donnelly_HFMN_AI_dataset_firstpass.xlsx   # Input dataset
```

## Input features

The model uses available numerical features from the dataset (missing features
are skipped automatically):

| Category    | Feature                        | Description                          |
|-------------|--------------------------------|--------------------------------------|
| Formulation | PVA_wt_pct                     | Poly(vinyl alcohol) concentration    |
| Formulation | PVP_wt_pct                     | Poly(vinyl pyrrolidone) concentration|
| Formulation | citric_acid_wt_pct             | Citric acid (crosslinker) concentration|
| Formulation | PMVE_MA_or_Gantrez_wt_pct      | Gantrez/PMVE-MA copolymer concentration|
| Formulation | PEG10000_wt_pct                | PEG 10,000 concentration             |
| Formulation | PEG400_wt_pct                  | PEG 400 concentration                |
| Formulation | glycerol_wt_pct                | Glycerol (plasticiser) concentration |
| Formulation | water_wt_pct                   | Water content                        |
| Process     | crosslink_temp_C               | Crosslinking temperature (°C)        |
| Process     | crosslink_time_min             | Crosslinking time (minutes)          |
| Geometry    | needle_number_total            | Number of needles per array          |
| Geometry    | needle_height_um               | Needle height (µm)                   |
| Geometry    | needle_base_width_um           | Needle base width (µm)               |
| Geometry    | interspacing_um                | Needle interspacing (µm)             |
| Geometry    | film_thickness_mm              | Backing film thickness (mm)          |

## Target properties

| Variable               | Description                                  |
|------------------------|----------------------------------------------|
| swelling_pct_mean      | Mean equilibrium swelling percentage         |
| gel_fraction_pct_mean  | Mean gel fraction percentage                 |
| Fmax_N_per_cm2_mean    | Mean maximum mechanical failure force (N/cm²)|

## Modelling approach

- **Missing values:** Median imputation for numerical input features.
- **Cross-validation:** Leave-One-Out (LOO) because the dataset has fewer than
  30 samples. Each sample is held out once; the model is trained on the
  remaining 24 and evaluated on the held-out sample.
- **Models compared:** Linear Regression and Random Forest Regressor
  (200 trees, max_depth=5 to limit overfitting).
- **Metrics:** R², MAE (mean absolute error), RMSE (root mean squared error).
- **Final model:** Random Forest — chosen because it handles small tabular
  datasets with nonlinear relationships better than linear models.
- **Minimum data requirement:** A target is only modelled if at least 5 valid
  rows are available; otherwise it is skipped with a clear message.

## Setup & usage

### 1. Install dependencies

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the models

```powershell
python train_model.py
```

This reads the Excel file, trains models, and writes outputs to `models/` and
`outputs/`.

### 3. Launch the Streamlit app

```powershell
streamlit run app.py
```

Open the URL printed in the terminal (usually http://localhost:8501).

Enter formulation, process, and geometry parameters in the sidebar and click
**Run Prediction** to see estimated properties.

## Limitations (please read)

1. **Small dataset:** The model is trained on ~25 data points extracted from
   the literature. Predictions carry high uncertainty.
2. **Narrow formulation space:** The training data covers a limited range of
   polymer concentrations and processing conditions. Predictions for
   formulations far outside these ranges are likely unreliable.
3. **No uncertainty quantification:** The app reports point predictions only.
   Confidence intervals are not provided because the small sample size makes
   them unreliable.
4. **Literature bias:** The dataset reflects published (successful)
   formulations. The model is blind to regions of the formulation space where
   microneedle fabrication fails.
5. **Use case:** This tool is intended for **early-stage screening** — e.g.,
   ranking candidate formulations for further experimental testing. Do not use
   it as a substitute for experimental measurement or as a go/no-go decision
   gate.

## License

This project is provided for academic/research use.
