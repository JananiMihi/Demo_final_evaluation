"""
extract_pca_loadings.py
Extracts PCA component loadings for HR and BP domains and saves to CSV.
"""
import warnings, os, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

WORK_DIR          = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(WORK_DIR, "Full_Data_Set_preprocessed.xlsx")
TARGET            = "volume_corrected_cm3"

# Load
df_raw = pd.read_excel(PREPROCESSED_PATH, sheet_name="preprocessed")
df     = df_raw.copy()
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

# Impute
for col in HR_FEATURES + BP_FEATURES:
    df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
    if df_main[col].isna().any():
        try:
            df_main[col] = df_main[col].fillna(df_main.groupby("Muscle")[col].transform("median"))
        except:
            pass
        df_main[col] = df_main[col].fillna(df_main[col].median())

# Log-transform
LOG_KW = ["abs","power","Total_power","RMSSD","SDNN","TINN","SD1","SD2","SampEn"]
for col in HR_FEATURES + BP_FEATURES:
    if col in df_main.columns and any(k in col for k in LOG_KW) and (df_main[col] > 0).all():
        df_main[col + "_log"] = np.log1p(df_main[col])

def use_log(fl, df):
    return list(dict.fromkeys([c+"_log" if c+"_log" in df.columns else c for c in fl]))

HR_FEAT = [c for c in use_log(HR_FEATURES, df_main) if c in df_main.columns]
BP_FEAT = [c for c in use_log(BP_FEATURES, df_main) if c in df_main.columns]

# PCA
scaler_hr = StandardScaler()
scaler_bp = StandardScaler()
X_hr = scaler_hr.fit_transform(df_main[HR_FEAT].values.astype(float))
X_bp = scaler_bp.fit_transform(df_main[BP_FEAT].values.astype(float))

def fit_pca(X, thresh=0.95):
    full = PCA().fit(X)
    cumv = np.cumsum(full.explained_variance_ratio_)
    n    = int(np.argmax(cumv >= thresh) + 1)
    pca  = PCA(n_components=n).fit(X)
    return pca, n

pca_hr, n_hr = fit_pca(X_hr)
pca_bp, n_bp = fit_pca(X_bp)

print(f"HR PCA: {n_hr} components | BP PCA: {n_bp} components")
print(f"HR features used: {len(HR_FEAT)}")
print(f"BP features used: {len(BP_FEAT)}")

# ── Save HR loadings ────────────────────────────────────────────────────────
hr_load = pd.DataFrame(
    pca_hr.components_.T,
    index=HR_FEAT,
    columns=[f"HR_PC{i+1}" for i in range(n_hr)]
)
hr_load.index.name = "Feature"
hr_load.to_csv(os.path.join(WORK_DIR, "pca_hr_loadings.csv"))
print(f"Saved pca_hr_loadings.csv  ({hr_load.shape})")

# ── Save BP loadings ────────────────────────────────────────────────────────
bp_load = pd.DataFrame(
    pca_bp.components_.T,
    index=BP_FEAT,
    columns=[f"BP_PC{i+1}" for i in range(n_bp)]
)
bp_load.index.name = "Feature"
bp_load.to_csv(os.path.join(WORK_DIR, "pca_bp_loadings.csv"))
print(f"Saved pca_bp_loadings.csv  ({bp_load.shape})")

# ── Save variance explained ─────────────────────────────────────────────────
var_df = pd.DataFrame({
    "PC":       [f"HR_PC{i+1}" for i in range(n_hr)],
    "Domain":   ["HR"] * n_hr,
    "VarExp_pct": np.round(pca_hr.explained_variance_ratio_ * 100, 4),
    "CumVar_pct": np.round(np.cumsum(pca_hr.explained_variance_ratio_) * 100, 4),
})
var_df_bp = pd.DataFrame({
    "PC":       [f"BP_PC{i+1}" for i in range(n_bp)],
    "Domain":   ["BP"] * n_bp,
    "VarExp_pct": np.round(pca_bp.explained_variance_ratio_ * 100, 4),
    "CumVar_pct": np.round(np.cumsum(pca_bp.explained_variance_ratio_) * 100, 4),
})
var_all = pd.concat([var_df, var_df_bp], ignore_index=True)
var_all.to_csv(os.path.join(WORK_DIR, "pca_variance_explained.csv"), index=False)
print(f"Saved pca_variance_explained.csv")

# ── Print top-5 loadings per first 3 PCs ───────────────────────────────────
print("\n── HR Domain: Top 5 absolute loadings per PC ──")
for pc in [f"HR_PC{i+1}" for i in range(min(5, n_hr))]:
    top = hr_load[pc].abs().nlargest(5)
    print(f"\n{pc}:")
    for feat, val in top.items():
        sign = "+" if hr_load.loc[feat, pc] > 0 else "-"
        print(f"  {sign}{abs(val):.4f}  {feat}")

print("\n── BP Domain: Top 5 absolute loadings per PC ──")
for pc in [f"BP_PC{i+1}" for i in range(min(4, n_bp))]:
    top = bp_load[pc].abs().nlargest(5)
    print(f"\n{pc}:")
    for feat, val in top.items():
        sign = "+" if bp_load.loc[feat, pc] > 0 else "-"
        print(f"  {sign}{abs(val):.4f}  {feat}")

print("\nDONE")
