"""
save_scaler_params.py
=====================
Run ONCE to export the StandardScaler mean/scale vectors and ordered feature
lists needed for projecting new RR-interval + BP data onto the trained PCA space.

Outputs (written to the same ML Modelling directory):
  pca_hr_scaler_mean.npy    shape (44,)   HR feature means
  pca_hr_scaler_scale.npy   shape (44,)   HR feature standard-deviations
  pca_bp_scaler_mean.npy    shape (19,)   BP feature means
  pca_bp_scaler_scale.npy   shape (19,)   BP feature standard-deviations
  pca_hr_features.json      ordered list of 44 log-transformed HR feature names
  pca_bp_features.json      ordered list of 19 (possibly log-transformed) BP feature names

Run:
    (venv) python save_scaler_params.py
"""

import warnings, json, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

WORK_DIR = Path(__file__).parent
PREPROCESSED = WORK_DIR / "Full_Data_Set_preprocessed.xlsx"
TARGET = "volume_corrected_cm3"

print("Loading preprocessed dataset …")
df_raw = pd.read_excel(PREPROCESSED, sheet_name="preprocessed")
df = df_raw.copy()
df["Group"]  = df["Group"].astype(str).str.strip()
df["Muscle"] = df["Muscle"].astype(str).str.strip()
df = df[df["Muscle"].notna() & (df["Muscle"] != "TR")].reset_index(drop=True)
df_main = df[df[TARGET].notna()].reset_index(drop=True)

HR_FEATURES = [
    "EX_RMSSD_ms","EX_SDNN_ms","EX_pNNxx_pct","EX_Mean_RR_ms","EX_Mean_HR_bpm",
    "EX_RR_tri_index","EX_TINN_ms",
    "EX_LF_abs_FFT","EX_HF_abs_FFT","EX_LF_HF_ratio_FFT","EX_VLF_abs_FFT",
    "EX_Total_power_FFT","EX_LF_nu_FFT","EX_HF_nu_FFT",
    "EX_SD1_ms","EX_SD2_ms","EX_SD2_SD1_ratio","EX_SampEn","EX_DFA_alpha1",
    "EX_RQA_RecurrenceRate","EX_RQA_Determinism",
    "REST_RMSSD_ms","REST_SDNN_ms","REST_pNNxx_pct","REST_Mean_RR_ms","REST_Mean_HR_bpm",
    "REST_LF_abs_FFT","REST_HF_abs_FFT","REST_LF_HF_ratio_FFT","REST_Total_power_FFT",
    "REST_LF_nu_FFT","REST_HF_nu_FFT",
    "REST_SD1_ms","REST_SD2_ms","REST_SampEn","REST_DFA_alpha1",
    "nabla_h_ex","nabla_h_rest",
    "GLOBAL_EX_RMSSD_ms","GLOBAL_EX_SDNN_ms","GLOBAL_REST_RMSSD_ms","GLOBAL_REST_SDNN_ms",
    "GLOBAL_nabla_h_ex","GLOBAL_nabla_h_rest",
]
BP_FEATURES = [
    "SBP_rest_pre_mmHg","DBP_rest_pre_mmHg","MAP_rest_pre_mmHg","PP_rest_pre_mmHg",
    "SBP_post_set_mmHg","DBP_post_set_mmHg","MAP_post_set_mmHg","delta_SBP_mmHg",
    "SBP_mean_e","DBP_mean_e","MAP_e","Delta_SBP_e","Delta_DBP_e",
    "ARV_SBP_mmHg","SV_SBP_mmHg","ARV_DBP_mmHg","SV_DBP_mmHg",
    "VIM_SBP_mmHg","VIM_DBP_mmHg",
]

HR_FEATURES = [c for c in HR_FEATURES if c in df_main.columns]
BP_FEATURES = [c for c in BP_FEATURES if c in df_main.columns]
print(f"HR features present: {len(HR_FEATURES)}  BP features present: {len(BP_FEATURES)}")

# Impute
for col in HR_FEATURES + BP_FEATURES:
    df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
    if df_main[col].isna().any():
        try:
            df_main[col] = df_main[col].fillna(
                df_main.groupby("Muscle")[col].transform("median"))
        except Exception:
            pass
        df_main[col] = df_main[col].fillna(df_main[col].median())

# Log-transform (same logic as extract_pca_loadings.py)
LOG_KW = ["abs","power","Total_power","RMSSD","SDNN","TINN","SD1","SD2","SampEn"]
for col in HR_FEATURES + BP_FEATURES:
    if col in df_main.columns and any(k in col for k in LOG_KW) and (df_main[col] > 0).all():
        df_main[col + "_log"] = np.log1p(df_main[col])

def use_log(fl, df):
    return list(dict.fromkeys(
        [c + "_log" if c + "_log" in df.columns else c for c in fl]
    ))

HR_FEAT = [c for c in use_log(HR_FEATURES, df_main) if c in df_main.columns]
BP_FEAT = [c for c in use_log(BP_FEATURES, df_main) if c in df_main.columns]
print(f"After log-transform — HR_FEAT: {len(HR_FEAT)}  BP_FEAT: {len(BP_FEAT)}")

# Fit scalers
scaler_hr = StandardScaler()
scaler_bp = StandardScaler()
X_hr = scaler_hr.fit_transform(df_main[HR_FEAT].values.astype(float))
X_bp = scaler_bp.fit_transform(df_main[BP_FEAT].values.astype(float))

# Verify PCA matches existing loadings
def fit_pca(X, thresh=0.95):
    full = PCA().fit(X)
    cumv = np.cumsum(full.explained_variance_ratio_)
    n    = int(np.argmax(cumv >= thresh) + 1)
    return PCA(n_components=n).fit(X), n

pca_hr, n_hr = fit_pca(X_hr)
pca_bp, n_bp = fit_pca(X_bp)
print(f"PCA components — HR: {n_hr}  BP: {n_bp}")

# ── Save ─────────────────────────────────────────────────────────────────────
np.save(WORK_DIR / "pca_hr_scaler_mean.npy",  scaler_hr.mean_)
np.save(WORK_DIR / "pca_hr_scaler_scale.npy", scaler_hr.scale_)
np.save(WORK_DIR / "pca_bp_scaler_mean.npy",  scaler_bp.mean_)
np.save(WORK_DIR / "pca_bp_scaler_scale.npy", scaler_bp.scale_)

with open(WORK_DIR / "pca_hr_features.json", "w") as f:
    json.dump(HR_FEAT, f, indent=2)
with open(WORK_DIR / "pca_bp_features.json", "w") as f:
    json.dump(BP_FEAT, f, indent=2)

print("\nSaved:")
print(f"  pca_hr_scaler_mean.npy  shape={scaler_hr.mean_.shape}")
print(f"  pca_hr_scaler_scale.npy shape={scaler_hr.scale_.shape}")
print(f"  pca_bp_scaler_mean.npy  shape={scaler_bp.mean_.shape}")
print(f"  pca_bp_scaler_scale.npy shape={scaler_bp.scale_.shape}")
print(f"  pca_hr_features.json    {len(HR_FEAT)} features")
print(f"  pca_bp_features.json    {len(BP_FEAT)} features")
print("\nDONE — scaler parameters saved.")
