"""
train_model.py
-------------
Trains regression models to predict hydrogel-forming microneedle (HFMN)
properties from formulation, process, and geometry variables.

Reads: Ryan_Donnelly_HFMN_AI_dataset_firstpass.xlsx (sheet: Extracted_Data)
Saves: models/*.joblib, outputs/*.xlsx, outputs/*.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 output on Windows consoles that default to a limited code page
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Ryan_Donnelly_HFMN_AI_dataset_firstpass.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHEET_NAME = "Extracted_Data"

# Candidate feature columns (only those present in the file will be used)
CANDIDATE_FEATURES = [
    "PVA_wt_pct",
    "PVP_wt_pct",
    "citric_acid_wt_pct",
    "PMVE_MA_or_Gantrez_wt_pct",
    "PEG10000_wt_pct",
    "PEG400_wt_pct",
    "glycerol_wt_pct",
    "water_wt_pct",
    "crosslink_temp_C",
    "crosslink_time_min",
    "needle_number_total",
    "needle_height_um",
    "needle_base_width_um",
    "interspacing_um",
    "film_thickness_mm",
]

TARGET_COLUMNS = [
    "swelling_pct_mean",
    "gel_fraction_pct_mean",
    "Fmax_N_per_cm2_mean",
]

TARGET_LABELS = {
    "swelling_pct_mean": "Swelling (%)",
    "gel_fraction_pct_mean": "Gel Fraction (%)",
    "Fmax_N_per_cm2_mean": "Fmax (N/cm^2)",
}

MIN_SAMPLES = 5
CV_THRESHOLD = 30  # <30 → LOO; >=30 → 5-fold

RANDOM_STATE = 42
RF_PARAMS = {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str, sheet: str) -> pd.DataFrame:
    """Load the Excel sheet."""
    df = pd.read_excel(path, sheet_name=sheet)
    # If the first column is a row counter, set it as index
    if df.columns[0] in ("row_id", "Row", "row"):
        df = df.set_index(df.columns[0])
    return df


def select_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidate columns present in df, excluding those that are all-NaN."""
    present = [c for c in candidates if c in df.columns]
    usable = [c for c in present if df[c].notna().sum() > 0]
    skipped_all_nan = [c for c in present if c not in usable]
    if skipped_all_nan:
        print(f"  Skipping all-NaN feature columns: {skipped_all_nan}")
    skipped_missing = [c for c in candidates if c not in df.columns]
    if skipped_missing:
        print(f"  Skipping missing feature columns (not in sheet): {skipped_missing}")
    return usable


def to_numeric_robust(series: pd.Series) -> pd.Series:
    """Coerce to numeric, handling stray strings like '>600'."""
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-eE]", "", regex=True), errors="coerce")


def evaluate_model(model, X: np.ndarray, y: np.ndarray, cv) -> dict[str, float]:
    """Run cross-validation via cross_val_predict and return R², MAE, RMSE."""
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return {
        "R2": r2_score(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
    }


def plot_actual_vs_predicted(y_true, y_pred, target_name: str, model_name: str, save_path: str):
    """Scatter plot of actual vs predicted values."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal fit")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{TARGET_LABELS.get(target_name, target_name)}\n{model_name} — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names: list[str], target_name: str, save_path: str):
    """Horizontal bar chart of feature importances (RF only)."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(7, max(3, len(feature_names) * 0.35)))
    plt.barh(range(len(idx)), imp[idx], align="center")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"{TARGET_LABELS.get(target_name, target_name)}\nFeature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("HFMN Machine Learning — Model Training")
    print("=" * 70)

    # 1. Load data
    print(f"\n[1/5] Loading data from {DATA_PATH}")
    df = load_data(DATA_PATH, SHEET_NAME)
    print(f"  {len(df)} rows × {len(df.columns)} columns")

    # 2. Select features
    print("\n[2/5] Selecting features")
    feature_cols = select_features(df, CANDIDATE_FEATURES)
    if not feature_cols:
        raise RuntimeError("No usable feature columns found — check the dataset.")
    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    # 3. Determine CV strategy
    cv = LeaveOneOut()
    print(f"\n  Total samples: {len(df)}  →  Using Leave-One-Out cross-validation (<{CV_THRESHOLD} rows)")

    # 4. Evaluate models for each target
    print("\n[3/5] Training & evaluating models for each target")
    performance_rows = []
    importance_rows = []
    all_models = {}

    for target in TARGET_COLUMNS:
        print(f"\n{'─' * 50}")
        print(f"  Target: {target}")

        if target not in df.columns:
            print(f"  → SKIP: column not found in dataset.")
            continue

        # Prepare target values
        y_raw = to_numeric_robust(df[target])
        mask = y_raw.notna()
        y = y_raw[mask].values.astype(float)
        X_raw = df.loc[mask, feature_cols].copy()

        if len(y) < MIN_SAMPLES:
            print(f"  → SKIP: only {len(y)} valid rows (< {MIN_SAMPLES} required).")
            continue

        # Impute missing features
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_raw)
        print(f"  Valid samples: {len(y)}  |  Features: {X.shape[1]}")

        # ---- Linear Regression ----
        print("  Training LinearRegression ...")
        lr = LinearRegression()
        lr_metrics = evaluate_model(lr, X, y, cv)
        print(f"    CV R2={lr_metrics['R2']:.4f}  MAE={lr_metrics['MAE']:.4f}  RMSE={lr_metrics['RMSE']:.4f}")

        # ---- Random Forest ----
        print("  Training RandomForestRegressor ...")
        rf = RandomForestRegressor(**RF_PARAMS)
        rf_metrics = evaluate_model(rf, X, y, cv)
        print(f"    CV R2={rf_metrics['R2']:.4f}  MAE={rf_metrics['MAE']:.4f}  RMSE={rf_metrics['RMSE']:.4f}")

        # Record performance
        for model_name, metrics in [("LinearRegression", lr_metrics), ("RandomForest", rf_metrics)]:
            performance_rows.append({
                "Target": target,
                "Model": model_name,
                "Samples": len(y),
                "Features": X.shape[1],
                "CV_Method": "LOO",
                **metrics,
            })

        # ---- Final RF model (fit on all data) ----
        print("  Fitting final RandomForest model on all data ...")
        final_rf = RandomForestRegressor(**RF_PARAMS)
        final_rf.fit(X, y)

        # Save model & metadata
        model_path = os.path.join(MODEL_DIR, f"model_{target}.joblib")
        joblib.dump({"model": final_rf, "imputer": imputer, "features": feature_cols}, model_path)
        all_models[target] = final_rf

        # Feature importance
        imp = final_rf.feature_importances_
        for feat, val in zip(feature_cols, imp):
            importance_rows.append({"Target": target, "Feature": feat, "Importance": val})

        # ---- Predictions on training data for plots ----
        y_pred = final_rf.predict(X)

        # Actual vs Predicted plot
        avp_path = os.path.join(OUTPUT_DIR, f"actual_vs_predicted_{target}.png")
        plot_actual_vs_predicted(y, y_pred, target, "Random Forest (final, all data)", avp_path)

        # Feature importance plot
        fi_path = os.path.join(OUTPUT_DIR, f"feature_importance_{target}.png")
        plot_feature_importance(final_rf, feature_cols, target, fi_path)

    # 5. Save outputs
    print(f"\n[4/5] Saving outputs to {OUTPUT_DIR}")
    perf_df = pd.DataFrame(performance_rows)
    perf_path = os.path.join(OUTPUT_DIR, "model_performance.xlsx")
    perf_df.to_excel(perf_path, index=False)
    print(f"  → {perf_path}")

    imp_df = pd.DataFrame(importance_rows)
    imp_path = os.path.join(OUTPUT_DIR, "feature_importance_each_target.xlsx")
    imp_df.to_excel(imp_path, index=False)
    print(f"  → {imp_path}")

    # Summary to console
    print(f"\n[5/5] Summary")
    print("=" * 70)
    print(perf_df.to_string(index=False))
    print("=" * 70)
    print(f"\nTrained models saved for: {list(all_models.keys())}")
    print(f"Model files: {os.listdir(MODEL_DIR)}")
    print("Done.")


if __name__ == "__main__":
    main()
