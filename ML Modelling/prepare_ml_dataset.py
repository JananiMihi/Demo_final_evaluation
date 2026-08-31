"""
prepare_ml_dataset.py
Builds the final, analysis-ready CSV for ML modelling.

Rows   : 234  (participants × muscles with valid volume measurements)
Columns: metadata  +  35 VIF-retained features  +  one-hot encodings  +  target

Output files
------------
ml_dataset_full.csv        -- all 234 rows, all features + target
ml_dataset_features.csv    -- features only (no target, no metadata) for quick inspection
ml_dataset_summary.csv     -- column-level statistics
"""

import warnings, os, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR           = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(OUT_DIR, "Full_Data_Set_preprocessed.xlsx")
TARGET            = "volume_corrected_cm3"

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load preprocessed dataset (already cleaned)
# ═══════════════════════════════════════════════════════════════
print("Step 1 — Loading preprocessed dataset ...")
df_raw = pd.read_excel(PREPROCESSED_PATH, sheet_name="preprocessed")
df     = df_raw.copy()
df["Group"]  = df["Group"].astype(str).str.strip()
df["Muscle"] = df["Muscle"].astype(str).str.strip()
df = df[df["Muscle"].notna() & (df["Muscle"] != "TR")].reset_index(drop=True)

df_main = df[df[TARGET].notna()].reset_index(drop=True)

# Gender_bin / Group_code already in preprocessed — recreate if missing
if "Gender_bin" not in df_main.columns:
    df_main["Gender_bin"] = (df_main["Gender"].astype(str).str.strip().str.lower() == "male").astype(int)
if "Group_code" not in df_main.columns:
    df_main["Group_code"] = df_main["Group"].map({"G1":1,"G2":2,"G3":3,"G4":4}).fillna(0).astype(int)

# shape_class already in preprocessed — recreate if missing
if "shape_class" not in df_main.columns or df_main["shape_class"].isna().all():
    SHAPE_MAP = {"BB":"Fusiform","TB":"Fusiform","DL":"Pennate","GA":"Pennate",
                 "FDS":"Unipennate","TA":"Unipennate","BF":"Strap","VL":"Strap"}
    df_main["shape_class"] = df_main["Muscle"].map(SHAPE_MAP)

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
COVARIATE_COLS = ["Age","Weight_kg","Height_cm","BMI","Gender_bin","Training_yrs",
                  "Stress_Index_pct","Sleep_Index_pct","Nutrition_Index_pct",
                  "Protein_g_kg","Calories_kcal","Group_code"]

HR_FEATURES    = [c for c in HR_FEATURES    if c in df_main.columns]
BP_FEATURES    = [c for c in BP_FEATURES    if c in df_main.columns]
COVARIATE_COLS = [c for c in COVARIATE_COLS if c in df_main.columns]

all_feat = HR_FEATURES + BP_FEATURES + COVARIATE_COLS + [TARGET]
for col in all_feat:
    df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
for col in all_feat:
    if df_main[col].isna().any():
        try:
            df_main[col] = df_main[col].fillna(df_main.groupby("Muscle")[col].transform("median"))
        except Exception:
            pass
        gmed = df_main[col].median()
        if pd.notna(gmed):
            df_main[col] = df_main[col].fillna(gmed)

LOG_KW = ["abs","power","Total_power","RMSSD","SDNN","TINN","SD1","SD2","SampEn"]
for col in HR_FEATURES + BP_FEATURES:
    if col in df_main.columns and any(k in col for k in LOG_KW) and (df_main[col] > 0).all():
        df_main[col + "_log"] = np.log1p(df_main[col])

def use_log(fl, df):
    return list(dict.fromkeys([c+"_log" if c+"_log" in df.columns else c for c in fl]))

HR_FEAT = [c for c in use_log(HR_FEATURES, df_main) if c in df_main.columns]
BP_FEAT = [c for c in use_log(BP_FEATURES, df_main) if c in df_main.columns]

print(f"  N = {len(df_main)} rows  |  {len(df_main['participant'].unique())} participants  |  {len(df_main['Muscle'].unique())} muscles")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Domain-wise PCA (identical to statistical_analysis.py)
# ═══════════════════════════════════════════════════════════════
print("Step 2 — Domain-wise PCA ...")
scaler_hr = StandardScaler()
scaler_bp = StandardScaler()
X_hr = scaler_hr.fit_transform(df_main[HR_FEAT].values.astype(float))
X_bp = scaler_bp.fit_transform(df_main[BP_FEAT].values.astype(float))

def fit_pca(X, thresh=0.95):
    full  = PCA().fit(X)
    cumv  = np.cumsum(full.explained_variance_ratio_)
    n     = int(np.argmax(cumv >= thresh) + 1)
    pca   = PCA(n_components=n).fit(X)
    return pca, pca.transform(X), n

pca_hr, scores_hr, n_hr = fit_pca(X_hr)
pca_bp, scores_bp, n_bp = fit_pca(X_bp)

pc_labels_hr = [f"HR_PC{i+1}" for i in range(n_hr)]
pc_labels_bp = [f"BP_PC{i+1}" for i in range(n_bp)]
pc_labels    = pc_labels_hr + pc_labels_bp
Z = np.hstack([scores_hr, scores_bp])

df_pca = pd.DataFrame(Z, columns=pc_labels)
df_pca["participant"] = df_main["participant"].values
df_pca["Muscle"]      = df_main["Muscle"].values
df_pca["Group"]       = df_main["Group"].values
df_pca["shape_class"] = df_main["shape_class"].values
df_pca[TARGET]        = df_main[TARGET].values
for cov in COVARIATE_COLS:
    if cov in df_main.columns:
        df_pca[cov] = df_main[cov].values

print(f"  HR: {n_hr} PCs  |  BP: {n_bp} PCs")

# ═══════════════════════════════════════════════════════════════
# STEP 3: VIF screening (identical to statistical_analysis.py)
# ═══════════════════════════════════════════════════════════════
print("Step 3 — VIF screening ...")
all_predictors = pc_labels + [c for c in COVARIATE_COLS if c in df_pca.columns]

def compute_vif(df_f, cols):
    cols = list(cols)
    while True:
        Xv   = df_f[cols].dropna().values.astype(float)
        Xm   = sm.add_constant(Xv, has_constant="add")
        vifs = {}
        for i, c in enumerate(cols):
            try:   vifs[c] = variance_inflation_factor(Xm, i+1)
            except: vifs[c] = np.nan
        worst = max(vifs, key=lambda c: vifs[c] if not np.isnan(vifs[c]) else 0)
        if np.isnan(vifs[worst]) or vifs[worst] <= 10.0:
            break
        cols.remove(worst)
    return cols, vifs

retained_cols, _ = compute_vif(df_pca, all_predictors)
retained_pcs  = [c for c in retained_cols if c in pc_labels]
retained_covs = [c for c in retained_cols if c not in pc_labels]
print(f"  Retained: {len(retained_cols)} features  ({len(retained_pcs)} PCs + {len(retained_covs)} covariates)")
print(f"  Retained PCs : {retained_pcs}")
print(f"  Retained covs: {retained_covs}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Build the ML dataset
# ═══════════════════════════════════════════════════════════════
print("Step 4 — Assembling ML dataset ...")

# ── 4a. Core dataframe: metadata + VIF-retained features + target ──────────
meta_cols = ["participant", "Group", "Muscle", "shape_class"]
ml_df = df_pca[meta_cols + retained_cols + [TARGET]].copy()

# ── 4b. Muscle one-hot encoding (for linear / DL models) ─────────────────
muscle_dummies = pd.get_dummies(ml_df["Muscle"], prefix="M", drop_first=False)
shape_dummies  = pd.get_dummies(ml_df["shape_class"], prefix="SC", drop_first=False)
group_dummies  = pd.get_dummies(ml_df["Group"],  prefix="G",  drop_first=False)

ml_df = pd.concat([ml_df, muscle_dummies, shape_dummies, group_dummies], axis=1)

# ── 4c. Integer label encodings (for tree-based models) ──────────────────
muscle_order = sorted(ml_df["Muscle"].unique())
shape_order  = ["Fusiform","Pennate","Unipennate","Strap"]
group_order  = ["G1","G2","G3","G4"]

ml_df["Muscle_label"]     = ml_df["Muscle"].map({m:i for i,m in enumerate(muscle_order)})
ml_df["shape_class_label"]= ml_df["shape_class"].map({s:i for i,s in enumerate(shape_order) if s in ml_df["shape_class"].unique()})
ml_df["Group_label"]      = ml_df["Group"].map({g:i for i,g in enumerate(group_order)})

# ── 4d. Final column order ─────────────────────────────────────────────────
# metadata | PC features | covariate features | one-hot encodings | label codes | target
pc_cols    = retained_pcs
cov_cols   = retained_covs
enc_cols   = (list(muscle_dummies.columns) + list(shape_dummies.columns) +
              list(group_dummies.columns) +
              ["Muscle_label","shape_class_label","Group_label"])
final_cols = meta_cols + pc_cols + cov_cols + enc_cols + [TARGET]
ml_df      = ml_df[final_cols]

# ── 4e. Verify no NaNs in feature columns ─────────────────────────────────
feat_cols = pc_cols + cov_cols + enc_cols + [TARGET]
nan_check = ml_df[feat_cols].isna().sum()
nan_cols  = nan_check[nan_check > 0]
if len(nan_cols):
    print(f"  WARNING: NaN values remain in: {nan_cols.to_dict()}")
else:
    print("  No NaN values in feature/target columns.")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Save outputs
# ═══════════════════════════════════════════════════════════════
full_path = os.path.join(OUT_DIR, "ml_dataset_full.csv")
ml_df.to_csv(full_path, index=False)

# Features-only (no metadata, no target)  — for sklearn pipelines
feat_only = ml_df[pc_cols + cov_cols + enc_cols]
feat_only.to_csv(os.path.join(OUT_DIR, "ml_dataset_features.csv"), index=False)

# Column metadata summary
summary_rows = []
for col in ml_df.columns:
    row_data = {"Column": col, "Type": str(ml_df[col].dtype),
                "Missing": int(ml_df[col].isna().sum()),
                "Min": None, "Max": None, "Mean": None, "Std": None}
    if pd.api.types.is_numeric_dtype(ml_df[col]):
        row_data.update({
            "Min":  round(float(ml_df[col].min()), 4),
            "Max":  round(float(ml_df[col].max()), 4),
            "Mean": round(float(ml_df[col].mean()), 4),
            "Std":  round(float(ml_df[col].std()),  4),
        })
    if col in meta_cols:    row_data["Role"] = "Metadata"
    elif col in pc_cols:    row_data["Role"] = "PC Feature (HR/BP)"
    elif col in cov_cols:   row_data["Role"] = "Covariate"
    elif col in enc_cols:   row_data["Role"] = "Encoding"
    elif col == TARGET:     row_data["Role"] = "Target"
    else:                   row_data["Role"] = "Other"
    summary_rows.append(row_data)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "ml_dataset_summary.csv"), index=False)

# ── Print report ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ML DATASET READY")
print("="*65)
print(f"  Rows          : {len(ml_df)}")
print(f"  Total columns : {len(ml_df.columns)}")
print()
print(f"  Metadata cols      : {len(meta_cols)}   {meta_cols}")
print(f"  PC features        : {len(pc_cols)}  {pc_cols}")
print(f"  Covariate features : {len(cov_cols)}  {cov_cols}")
print(f"  One-hot / label enc: {len(enc_cols)}")
print(f"  Target             : {TARGET}")
print()
print("  Groups:")
for g, cnt in ml_df["Group"].value_counts().sort_index().items():
    print(f"    {g}: {cnt} rows")
print()
print("  Muscles:")
for m, cnt in ml_df["Muscle"].value_counts().sort_index().items():
    print(f"    {m}: {cnt} rows")
print()
print("  Target statistics:")
print(f"    Mean  : {ml_df[TARGET].mean():.2f} cm3")
print(f"    Std   : {ml_df[TARGET].std():.2f} cm3")
print(f"    Min   : {ml_df[TARGET].min():.2f} cm3")
print(f"    Max   : {ml_df[TARGET].max():.2f} cm3")
print()
print("Files saved:")
for fname in ["ml_dataset_full.csv","ml_dataset_features.csv","ml_dataset_summary.csv"]:
    path = os.path.join(OUT_DIR, fname)
    sz   = os.path.getsize(path) // 1024
    print(f"  {fname:40s}  {sz:>4d} KB  ({len(open(path).readlines())-1} data rows)")
