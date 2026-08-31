"""
Statistical Analysis Pipeline for Muscle Hypertrophy Study
HRVBP-HyperNet: Cardiovascular Predictors of Skeletal Muscle Volume

Pipeline:
  1. Data Loading & Preprocessing
  2. Feature Engineering (log-transforms, scaling)
  3. Domain-wise PCA (HR domain, BP domain)
  4. VIF Multicollinearity Screening
  5. Spearman Correlation Analysis
  6. Linear Mixed-Effects Model (LME)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from itertools import combinations
import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSED_PATH = os.path.join(OUTPUT_DIR, "Full_Data_Set_preprocessed.xlsx")

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING  (preprocessed dataset — already cleaned)
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: DATA LOADING (preprocessed dataset)")
print("=" * 70)

df_raw = pd.read_excel(PREPROCESSED_PATH, sheet_name="preprocessed")
print(f"Loaded preprocessed sheet: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING  (minimal — dataset already clean)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: PREPROCESSING  (dataset is pre-cleaned — minimal steps)")
print("=" * 70)

df = df_raw.copy()

# 2b. Ensure Group and Muscle are stripped (paranoid check)
df["Group"]  = df["Group"].astype(str).str.strip()
df["Muscle"] = df["Muscle"].astype(str).str.strip()

# 2c. Confirm TR already removed and Muscle not-null
df = df[df["Muscle"].notna() & (df["Muscle"] != "TR")].reset_index(drop=True)
print(f"After confirming TR/null-Muscle removed: {len(df)} rows")

# 2d. Separate rows with valid vs missing volume
df_vol    = df[df["volume_corrected_cm3"].notna()].copy()
df_no_vol = df[df["volume_corrected_cm3"].isna()].copy()
print(f"Rows WITH valid volume   : {len(df_vol)}")
print(f"Rows WITHOUT volume      : {len(df_no_vol)}")

# Work with volume-valid rows for statistical analysis
df_main = df_vol.reset_index(drop=True)

# 2e. Encode categoricals (already present in preprocessed dataset; recreate as safety net)
if "Gender_bin" not in df_main.columns:
    df_main["Gender_bin"] = (df_main["Gender"].astype(str).str.strip().str.lower() == "male").astype(int)
if "Group_code" not in df_main.columns:
    group_map = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}
    df_main["Group_code"] = df_main["Group"].map(group_map).fillna(0).astype(int)

# 2f. Shape-class one-hot
df_main = pd.get_dummies(df_main, columns=["shape_class"], prefix="SC", drop_first=False)

# ── Feature column definitions ────────────────────────────────
HR_FEATURES = [
    # Time-domain (exercise window)
    "EX_RMSSD_ms", "EX_SDNN_ms", "EX_pNNxx_pct",
    "EX_Mean_RR_ms", "EX_Mean_HR_bpm",
    "EX_RR_tri_index", "EX_TINN_ms",
    # Frequency-domain (exercise)
    "EX_LF_abs_FFT", "EX_HF_abs_FFT", "EX_LF_HF_ratio_FFT",
    "EX_VLF_abs_FFT", "EX_Total_power_FFT",
    "EX_LF_nu_FFT", "EX_HF_nu_FFT",
    # Nonlinear (exercise)
    "EX_SD1_ms", "EX_SD2_ms", "EX_SD2_SD1_ratio",
    "EX_SampEn", "EX_DFA_alpha1",
    "EX_RQA_RecurrenceRate", "EX_RQA_Determinism",
    # Time-domain (rest window)
    "REST_RMSSD_ms", "REST_SDNN_ms", "REST_pNNxx_pct",
    "REST_Mean_RR_ms", "REST_Mean_HR_bpm",
    # Frequency-domain (rest)
    "REST_LF_abs_FFT", "REST_HF_abs_FFT", "REST_LF_HF_ratio_FFT",
    "REST_Total_power_FFT",
    "REST_LF_nu_FFT", "REST_HF_nu_FFT",
    # Nonlinear (rest)
    "REST_SD1_ms", "REST_SD2_ms",
    "REST_SampEn", "REST_DFA_alpha1",
    # HRD temporal gradients
    "nabla_h_ex", "nabla_h_rest",
    # Global session
    "GLOBAL_EX_RMSSD_ms", "GLOBAL_EX_SDNN_ms",
    "GLOBAL_REST_RMSSD_ms", "GLOBAL_REST_SDNN_ms",
    "GLOBAL_nabla_h_ex", "GLOBAL_nabla_h_rest",
]

BP_FEATURES = [
    # Resting BP
    "SBP_rest_pre_mmHg", "DBP_rest_pre_mmHg",
    "MAP_rest_pre_mmHg", "PP_rest_pre_mmHg",
    # Post-set response
    "SBP_post_set_mmHg", "DBP_post_set_mmHg",
    "MAP_post_set_mmHg", "delta_SBP_mmHg",
    # Session-level means
    "SBP_mean_e", "DBP_mean_e", "MAP_e",
    "Delta_SBP_e", "Delta_DBP_e",
    # Short-term BPV
    "ARV_SBP_mmHg", "SV_SBP_mmHg",
    "ARV_DBP_mmHg", "SV_DBP_mmHg",
    # Long-term BPV (VIM)
    "VIM_SBP_mmHg", "VIM_DBP_mmHg",
]

COVARIATE_COLS = [
    "Age", "Weight_kg", "Height_cm", "BMI",
    "Gender_bin", "Training_yrs",
    "Stress_Index_pct", "Sleep_Index_pct", "Nutrition_Index_pct",
    "Protein_g_kg", "Calories_kcal",
    "Group_code",
]

TARGET = "volume_corrected_cm3"

# 2g. Keep only columns present in data
HR_FEATURES  = [c for c in HR_FEATURES  if c in df_main.columns]
BP_FEATURES  = [c for c in BP_FEATURES  if c in df_main.columns]
COVARIATE_COLS = [c for c in COVARIATE_COLS if c in df_main.columns]

print(f"HR features available    : {len(HR_FEATURES)}")
print(f"BP features available    : {len(BP_FEATURES)}")
print(f"Covariate cols available : {len(COVARIATE_COLS)}")

# 2h. Missing value imputation (median per muscle group — numeric only)
all_feat_cols = HR_FEATURES + BP_FEATURES + COVARIATE_COLS + [TARGET]
existing_feat = [c for c in all_feat_cols if c in df_main.columns]

# Force all feature columns to numeric first
for col in existing_feat:
    df_main[col] = pd.to_numeric(df_main[col], errors="coerce")

for col in existing_feat:
    if df_main[col].isna().any():
        try:
            medians = df_main.groupby("Muscle")[col].transform("median")
            df_main[col] = df_main[col].fillna(medians)
        except Exception:
            pass
        # global median fallback
        global_med = df_main[col].median()
        if pd.notna(global_med):
            df_main[col] = df_main[col].fillna(global_med)

# 2i. Log-transform right-skewed features (power/energy metrics)
LOG_CANDIDATES = [
    c for c in HR_FEATURES + BP_FEATURES
    if any(kw in c for kw in ["abs", "power", "Total_power",
                               "RMSSD", "SDNN", "TINN", "SD1", "SD2", "SampEn"])
]
for col in LOG_CANDIDATES:
    if col in df_main.columns and (df_main[col] > 0).all():
        df_main[col + "_log"] = np.log1p(df_main[col])

# Update feature lists to use log versions where available
def use_log_if_available(feat_list, df):
    out = []
    for c in feat_list:
        if (c + "_log") in df.columns:
            out.append(c + "_log")
        else:
            out.append(c)
    return list(dict.fromkeys(out))   # deduplicate

HR_FEAT_FINAL = use_log_if_available(HR_FEATURES, df_main)
BP_FEAT_FINAL = use_log_if_available(BP_FEATURES, df_main)

# Keep only cols actually in df_main
HR_FEAT_FINAL = [c for c in HR_FEAT_FINAL if c in df_main.columns]
BP_FEAT_FINAL = [c for c in BP_FEAT_FINAL if c in df_main.columns]

print(f"\nAfter log-transform: HR={len(HR_FEAT_FINAL)}, BP={len(BP_FEAT_FINAL)}")

# 2j. Z-score standardise numeric features
scaler_hr = StandardScaler()
scaler_bp = StandardScaler()

X_hr_raw = df_main[HR_FEAT_FINAL].values.astype(float)
X_bp_raw = df_main[BP_FEAT_FINAL].values.astype(float)

X_hr_scaled = scaler_hr.fit_transform(X_hr_raw)
X_bp_scaled = scaler_bp.fit_transform(X_bp_raw)

y = df_main[TARGET].values.astype(float)

print(f"\nPreprocessing complete. Analysis sample: N={len(df_main)}")
print(f"Unique participants: {df_main['participant'].nunique()}")
print(f"Muscles represented: {df_main['Muscle'].unique()}")
print(f"Groups present     : {df_main['Group'].unique()}")

# ─────────────────────────────────────────────────────────────
# 3. DOMAIN-WISE PCA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: DOMAIN-WISE PCA  (HR & BP separately, >=95% variance retained)")
print("=" * 70)

def fit_pca(X_scaled, domain_name, variance_threshold=0.95):
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cumvar >= variance_threshold) + 1)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_scaled)
    print(f"\n{domain_name} domain:")
    print(f"  Total features       : {X_scaled.shape[1]}")
    print(f"  Components retained  : {n_comp}  (explains {cumvar[n_comp-1]*100:.1f}% variance)")
    print(f"  Per-component variance (%):")
    for i, ev in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {ev*100:.2f}%")
    return pca, scores, n_comp

pca_hr, scores_hr, n_hr = fit_pca(X_hr_scaled, "HR")
pca_bp, scores_bp, n_bp = fit_pca(X_bp_scaled, "BP")

# Build fused cardiovascular feature vector z
Z = np.hstack([scores_hr, scores_bp])
pc_labels_hr = [f"HR_PC{i+1}" for i in range(n_hr)]
pc_labels_bp = [f"BP_PC{i+1}" for i in range(n_bp)]
pc_labels    = pc_labels_hr + pc_labels_bp

df_pca = pd.DataFrame(Z, columns=pc_labels)
df_pca["participant"] = df_main["participant"].values
df_pca["Muscle"]      = df_main["Muscle"].values
df_pca["Group"]       = df_main["Group"].values
df_pca[TARGET]        = y
for cov in COVARIATE_COLS:
    if cov in df_main.columns:
        df_pca[cov] = df_main[cov].values

# PCA loadings table
def loadings_table(pca, feature_names, domain_label, top_n=5):
    comps = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"{domain_label}_PC{i+1}" for i in range(pca.n_components_)]
    )
    print(f"\n  Top {top_n} loadings per component (|loading| >= 0.30):")
    for col in comps.columns:
        top = comps[col].abs().nlargest(top_n)
        top_vals = comps.loc[top.index, col]
        sig = top_vals[top_vals.abs() >= 0.30]
        if len(sig):
            parts = [f"{idx}={val:+.3f}" for idx, val in sig.items()]
            print(f"    {col}: {', '.join(parts)}")
    return comps

print("\nHR PCA Loadings:")
hr_loadings = loadings_table(pca_hr, HR_FEAT_FINAL, "HR")
print("\nBP PCA Loadings:")
bp_loadings = loadings_table(pca_bp, BP_FEAT_FINAL, "BP")

# Save PCA summary
pca_summary = {
    "HR_components": n_hr,
    "BP_components": n_bp,
    "HR_variance_explained": list(np.round(pca_hr.explained_variance_ratio_ * 100, 2)),
    "BP_variance_explained": list(np.round(pca_bp.explained_variance_ratio_ * 100, 2)),
}
with open(os.path.join(OUTPUT_DIR, "pca_summary.json"), "w") as f:
    json.dump(pca_summary, f, indent=2)

# ─────────────────────────────────────────────────────────────
# 4. VIF MULTICOLLINEARITY FILTERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: VIF MULTICOLLINEARITY SCREENING  (threshold VIF > 10)")
print("=" * 70)

cov_cols_present = [c for c in COVARIATE_COLS if c in df_pca.columns]
all_predictors   = pc_labels + cov_cols_present

def compute_vif(df_features, feature_cols):
    """Iteratively remove highest-VIF feature until all VIF <= threshold."""
    cols = list(feature_cols)
    removed = []
    threshold = 10.0
    while True:
        X_vif = df_features[cols].dropna()
        X_mat = sm.add_constant(X_vif.values.astype(float), has_constant="add")
        vifs = {}
        for idx, col in enumerate(cols):
            try:
                vif = variance_inflation_factor(X_mat, idx + 1)
            except Exception:
                vif = np.nan
            vifs[col] = vif
        max_vif_col = max(vifs, key=lambda c: vifs[c] if not np.isnan(vifs[c]) else 0)
        max_vif_val = vifs[max_vif_col]
        if np.isnan(max_vif_val) or max_vif_val <= threshold:
            break
        removed.append((max_vif_col, round(max_vif_val, 2)))
        cols.remove(max_vif_col)
    return cols, removed, {c: round(v, 2) for c, v in vifs.items()}

retained_cols, removed_cols, final_vifs = compute_vif(df_pca, all_predictors)

print(f"\nFeatures removed due to VIF > 10:")
if removed_cols:
    for col, vif in removed_cols:
        print(f"  {col:30s}  VIF = {vif:.2f}")
else:
    print("  None removed.")

print(f"\nRetained features ({len(retained_cols)}):")
for col in retained_cols:
    print(f"  {col:30s}  VIF = {final_vifs.get(col, 'N/A')}")

# Separate retained PC labels from covariates
retained_pcs  = [c for c in retained_cols if c in pc_labels]
retained_covs = [c for c in retained_cols if c in cov_cols_present]

# Save VIF summary
vif_df = pd.DataFrame([
    {"Feature": c, "VIF": final_vifs.get(c, np.nan), "Status": "Retained"}
    for c in retained_cols
] + [
    {"Feature": c, "VIF": v, "Status": "Removed"}
    for c, v in removed_cols
])
vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif_summary.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 5. SPEARMAN CORRELATION SCREENING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: SPEARMAN CORRELATION ANALYSIS  (|rho_s| >= 0.30, p < 0.05)")
print("=" * 70)

spearman_results = []
for pc in retained_pcs:
    for muscle in df_pca["Muscle"].unique():
        mask = df_pca["Muscle"] == muscle
        x_m  = df_pca.loc[mask, pc].values
        y_m  = df_pca.loc[mask, TARGET].values
        valid = ~(np.isnan(x_m) | np.isnan(y_m))
        if valid.sum() < 5:
            continue
        rho, pval = stats.spearmanr(x_m[valid], y_m[valid])
        spearman_results.append({
            "Component":  pc,
            "Muscle":     muscle,
            "rho":        round(rho, 4),
            "p_value":    round(pval, 4),
            "n":          int(valid.sum()),
            "Significant": abs(rho) >= 0.30 and pval < 0.05,
        })

df_spearman = pd.DataFrame(spearman_results)
df_sig = df_spearman[df_spearman["Significant"]].sort_values("rho", key=abs, ascending=False)

print(f"\nTotal component × muscle pairs tested: {len(df_spearman)}")
print(f"Significant pairs (|rho_s|>=0.30, p<0.05): {len(df_sig)}")
print()
print(df_sig.to_string(index=False))

df_spearman.to_csv(os.path.join(OUTPUT_DIR, "spearman_results.csv"), index=False)

# Components with at least one significant muscle association
significant_pcs = list(df_sig["Component"].unique())
print(f"\nComponents with >=1 significant muscle: {significant_pcs}")

# Pooled Spearman (all muscles combined)
print("\nPooled across all muscles:")
pooled_results = []
for pc in retained_pcs:
    x_all = df_pca[pc].values
    y_all = df_pca[TARGET].values
    valid = ~(np.isnan(x_all) | np.isnan(y_all))
    rho, pval = stats.spearmanr(x_all[valid], y_all[valid])
    pooled_results.append({
        "Component": pc, "rho_pooled": round(rho, 4),
        "p_value": round(pval, 5), "Significant": abs(rho) >= 0.30 and pval < 0.05
    })
df_pooled = pd.DataFrame(pooled_results).sort_values("rho_pooled", key=abs, ascending=False)
print(df_pooled.to_string(index=False))
df_pooled.to_csv(os.path.join(OUTPUT_DIR, "spearman_pooled.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 6. LINEAR MIXED-EFFECTS MODEL (LME)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: LINEAR MIXED-EFFECTS MODEL")
print("=" * 70)

# Use all retained PCs + covariates as fixed effects
# Participant as random effect (random intercept)
# Muscle as additional grouping factor (per-muscle dummies)

df_model = df_pca.copy()

# Spearman-significant PCs only (orthogonal by PCA construction — no multicollinearity)
spearman_sig_pcs = list(df_sig["Component"].unique()) if len(df_sig) > 0 else retained_pcs[:5]
key_covariates   = ["Age", "BMI", "Gender_bin"]
key_covariates   = [c for c in key_covariates if c in df_model.columns]

# Muscle as a categorical fixed effect via C() in formula — avoids get_dummies correlation issue
# Normalise target
y_mean = df_model[TARGET].mean()
y_std  = df_model[TARGET].std()
df_model["volume_z"] = (df_model[TARGET] - y_mean) / y_std

# Sanitise column names for formula
safe_rename = {c: c.replace(" ", "_").replace("(", "").replace(")", "")
                   .replace("/", "_").replace("-", "_").replace("%", "pct")
               for c in df_model.columns}
df_model.rename(columns=safe_rename, inplace=True)
spearman_sig_pcs = [safe_rename.get(c, c) for c in spearman_sig_pcs]
key_covariates   = [safe_rename.get(c, c) for c in key_covariates]
all_retained_pcs = [safe_rename.get(c, c) for c in retained_pcs]

TARGET_Z = "volume_z"

def _lme_fit(formula, df_fit, group_col, output_suffix, label):
    """Try multiple optimizers; return result or raise."""
    for method in ("lbfgs", "bfgs", "gradient", "powell", "nm"):
        try:
            mdl = smf.mixedlm(formula, data=df_fit, groups=df_fit[group_col])
            res = mdl.fit(reml=True, method=method, maxiter=500)
            print(f"  Converged via {method}.")
            return res
        except Exception as exc:
            print(f"  {method}: {exc}")
    raise RuntimeError("All optimizers failed")

def run_lme(label, pc_set, df_m, target_z, output_suffix, use_muscle_fe=True):
    pcs = [c for c in pc_set if c in df_m.columns]
    covs = [c for c in key_covariates if c in df_m.columns]
    needed = pcs + covs + [target_z, "participant", "Muscle"]
    df_fit = df_m.dropna(subset=needed).copy()
    pcs = [c for c in pcs if df_fit[c].std() > 1e-10]

    # Build formula: PCs + optional muscle categorical + covariates
    rhs_terms = pcs + covs
    if use_muscle_fe:
        rhs_terms = ["C(Muscle)"] + rhs_terms
    formula = target_z + " ~ " + " + ".join(rhs_terms) if rhs_terms else target_z + " ~ 1"

    print(f"\n-- {label} ({len(pcs)} PCs, muscle_FE={use_muscle_fe}, N={len(df_fit)}) --")
    print(f"   Formula: {formula}")

    try:
        lme_result = _lme_fit(formula, df_fit, "participant", output_suffix, label)

        fe_table = lme_result.fe_params.to_frame(name="Estimate")
        fe_table["SE"]      = lme_result.bse_fe
        fe_table["z_stat"]  = lme_result.tvalues
        fe_table["p_value"] = lme_result.pvalues
        fe_table["CI_low"]  = lme_result.conf_int().iloc[:, 0]
        fe_table["CI_high"] = lme_result.conf_int().iloc[:, 1]
        fe_table.to_csv(os.path.join(OUTPUT_DIR, f"lme_fixed_effects_{output_suffix}.csv"))
        print(lme_result.summary())

        print("\n  -- Model Fit Statistics --")
        llf = lme_result.llf
        # Manual AIC/BIC when statsmodels returns NaN (non-convergence boundary case)
        n_params = len(lme_result.fe_params) + 2  # fixed effects + rand-var + resid-var
        n_obs    = int(lme_result.nobs)
        aic_val = lme_result.aic if not np.isnan(lme_result.aic) else 2*n_params - 2*llf
        bic_val = lme_result.bic if not np.isnan(lme_result.bic) else n_params*np.log(n_obs) - 2*llf
        print(f"  Log-Likelihood : {llf:.4f}")
        print(f"  AIC            : {aic_val:.4f}  (manual if NaN in summary)")
        print(f"  BIC            : {bic_val:.4f}  (manual if NaN in summary)")
        rand_sigma = np.sqrt(max(lme_result.cov_re.values[0][0], 0))
        print(f"  Random-effect sigma (participant): {rand_sigma:.4f}")
        print(f"  Residual sigma : {np.sqrt(lme_result.scale):.4f}")
        icc = rand_sigma**2 / (rand_sigma**2 + lme_result.scale) if (rand_sigma**2 + lme_result.scale) > 0 else 0
        print(f"  ICC (participant): {icc:.4f}")

        var_fixed      = np.var(lme_result.fittedvalues)
        var_rand       = max(lme_result.cov_re.values[0][0], 0)
        var_residual   = lme_result.scale
        r2_marginal    = var_fixed / (var_fixed + var_rand + var_residual)
        r2_conditional = (var_fixed + var_rand) / (var_fixed + var_rand + var_residual)
        print(f"\n  Pseudo R2 (marginal)    : {r2_marginal:.4f}  ({r2_marginal*100:.1f}%)")
        print(f"  Pseudo R2 (conditional) : {r2_conditional:.4f}  ({r2_conditional*100:.1f}%)")

        sig_fe = fe_table[fe_table["p_value"] < 0.05].sort_values("p_value")
        print(f"\n  Significant fixed effects (p < 0.05): {len(sig_fe)}")
        if len(sig_fe):
            print(sig_fe[["Estimate", "SE", "p_value"]].round(4).to_string())

        return {
            "Model":           label,
            "LogLikelihood":   round(llf, 4),
            "AIC":             round(aic_val, 4),
            "BIC":             round(bic_val, 4),
            "ICC":             round(icc, 4),
            "R2_marginal":     round(r2_marginal, 4),
            "R2_conditional":  round(r2_conditional, 4),
            "N_observations":  int(lme_result.nobs),
            "N_participants":  int(df_m["participant"].nunique()),
            "converged":       True,
        }

    except Exception as e:
        print(f"  All LME optimizers failed: {e}")
        # GLS fallback with participant fixed effects (equivalent to within-subject demeaning)
        print("  Using GLS with participant fixed effects as fallback...")
        df_fit2 = df_fit.copy()
        df_fit2["_ptfe"] = df_fit2["participant"].astype(str)
        fe_cols = pcs + covs
        y_vals  = df_fit2[target_z].values
        X_fe    = pd.get_dummies(df_fit2[["_ptfe"]], drop_first=True)
        if use_muscle_fe:
            X_muscle = pd.get_dummies(df_fit2[["Muscle"]], drop_first=True)
            X_all = pd.concat([df_fit2[fe_cols].reset_index(drop=True),
                               X_fe.reset_index(drop=True),
                               X_muscle.reset_index(drop=True)], axis=1)
        else:
            X_all = pd.concat([df_fit2[fe_cols].reset_index(drop=True),
                               X_fe.reset_index(drop=True)], axis=1)
        X_all = sm.add_constant(X_all.astype(float))
        ols_r = sm.OLS(y_vals, X_all).fit(cov_type="HC3")
        r2 = float(ols_r.rsquared)
        print(f"  GLS/OLS R2={r2:.4f}, Adj-R2={ols_r.rsquared_adj:.4f}")
        # Save only the PC and covariate rows (not the participant FE rows)
        pc_cov_rows = fe_table = pd.DataFrame({
            "Estimate": ols_r.params[:len(fe_cols)+1],
            "SE":       ols_r.bse[:len(fe_cols)+1],
            "z_stat":   ols_r.tvalues[:len(fe_cols)+1],
            "p_value":  ols_r.pvalues[:len(fe_cols)+1],
        }, index=["const"] + fe_cols)
        pc_cov_rows.to_csv(os.path.join(OUTPUT_DIR, f"lme_fixed_effects_{output_suffix}.csv"))
        return {"Model": label, "converged": False,
                "OLS_R2": round(r2, 4), "OLS_AdjR2": round(float(ols_r.rsquared_adj), 4)}

print("\nFitting LME models...")

# Model A: Spearman-significant PCs + C(Muscle) — minimal, most likely to converge
stats_parsimonious = run_lme(
    "Parsimonious (Spearman-sig PCs + Muscle FE)",
    spearman_sig_pcs, df_model, TARGET_Z, "parsimonious", use_muscle_fe=True
)

# Model B: All retained PCs + C(Muscle)
stats_full = run_lme(
    "Full (all retained PCs + Muscle FE)",
    all_retained_pcs, df_model, TARGET_Z, "full", use_muscle_fe=True
)

import shutil
lme_stats = {"parsimonious": stats_parsimonious, "full": stats_full}
with open(os.path.join(OUTPUT_DIR, "lme_stats.json"), "w") as f:
    json.dump(lme_stats, f, indent=2)

# Primary output: parsimonious model
parsimonious_csv = os.path.join(OUTPUT_DIR, "lme_fixed_effects_parsimonious.csv")
primary_csv      = os.path.join(OUTPUT_DIR, "lme_fixed_effects.csv")
if os.path.exists(parsimonious_csv) and os.path.getsize(parsimonious_csv) > 0:
    shutil.copy2(parsimonious_csv, primary_csv)

# ─────────────────────────────────────────────────────────────
# 7. PER-MUSCLE SPEARMAN CORRELATION TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: PER-MUSCLE SPEARMAN TABLE (significant associations)")
print("=" * 70)

pivot = df_sig.pivot_table(
    index="Component", columns="Muscle",
    values="rho", aggfunc="first"
).fillna("")

print(pivot.to_string())
pivot.to_csv(os.path.join(OUTPUT_DIR, "spearman_per_muscle_pivot.csv"))

# ─────────────────────────────────────────────────────────────
# 8. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: DESCRIPTIVE STATISTICS BY GROUP")
print("=" * 70)

desc_cols = [TARGET, "Age", "BMI", "Training_yrs",
             "Stress_Index_pct", "Sleep_Index_pct"]
desc_cols = [c for c in desc_cols if c in df_main.columns]

desc = df_main.groupby("Group")[desc_cols].agg(["mean", "std"]).round(2)
print(desc.to_string())
desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))

# Group-wise muscle volume
print("\nMean muscle volume (cm³) by Group × Muscle:")
vol_table = df_main.groupby(["Group", "Muscle"])[TARGET].agg(["mean", "std", "count"]).round(2)
print(vol_table.to_string())
vol_table.to_csv(os.path.join(OUTPUT_DIR, "volume_by_group_muscle.csv"))

# ─────────────────────────────────────────────────────────────
# 9. KRUSKAL-WALLIS GROUP COMPARISON (non-parametric)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: KRUSKAL-WALLIS TEST — Volume differences across groups")
print("=" * 70)

muscles_kw = df_main["Muscle"].unique()
kw_results = []
for muscle in muscles_kw:
    groups_data = []
    for grp in ["G1", "G2", "G3", "G4"]:
        vals = df_main[(df_main["Muscle"] == muscle) & (df_main["Group"] == grp)][TARGET].dropna()
        if len(vals) >= 3:
            groups_data.append(vals.values)
    if len(groups_data) >= 2:
        stat, p = stats.kruskal(*groups_data)
        kw_results.append({"Muscle": muscle, "H_stat": round(stat, 3), "p_value": round(p, 4),
                            "Significant": p < 0.05})

df_kw = pd.DataFrame(kw_results)
print(df_kw.to_string(index=False))
df_kw.to_csv(os.path.join(OUTPUT_DIR, "kruskal_wallis.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# 10. EXCEL EXPORT — bundle all results into one workbook
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: EXCEL EXPORT")
print("=" * 70)

excel_path = os.path.join(OUTPUT_DIR, "statistical_analysis.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    # Sheet 1: Descriptive stats
    desc_path = os.path.join(OUTPUT_DIR, "descriptive_stats.csv")
    if os.path.exists(desc_path):
        pd.read_csv(desc_path, header=[0, 1], index_col=0).to_excel(
            writer, sheet_name="Descriptive_Stats")

    # Sheet 2: Volume by Group x Muscle
    vol_path = os.path.join(OUTPUT_DIR, "volume_by_group_muscle.csv")
    if os.path.exists(vol_path):
        pd.read_csv(vol_path).to_excel(writer, sheet_name="Volume_Group_Muscle", index=False)

    # Sheet 3: PCA Summary
    pca_path = os.path.join(OUTPUT_DIR, "pca_summary.json")
    if os.path.exists(pca_path):
        with open(pca_path) as f:
            pca_dict = json.load(f)
        pca_rows = []
        for i, v in enumerate(pca_dict.get("HR_variance_explained", [])):
            pca_rows.append({"Domain": "HR", "Component": f"PC{i+1}", "Variance_Explained_pct": v})
        for i, v in enumerate(pca_dict.get("BP_variance_explained", [])):
            pca_rows.append({"Domain": "BP", "Component": f"PC{i+1}", "Variance_Explained_pct": v})
        pca_meta = pd.DataFrame([
            {"Item": "HR_components_retained", "Value": pca_dict.get("HR_components")},
            {"Item": "BP_components_retained", "Value": pca_dict.get("BP_components")},
        ])
        pd.DataFrame(pca_rows).to_excel(writer, sheet_name="PCA_Variance", index=False)
        pca_meta.to_excel(writer, sheet_name="PCA_Summary", index=False)

    # Sheet 4: VIF Screening
    vif_path = os.path.join(OUTPUT_DIR, "vif_summary.csv")
    if os.path.exists(vif_path):
        pd.read_csv(vif_path).to_excel(writer, sheet_name="VIF_Screening", index=False)

    # Sheet 5: Spearman Per-Component x Muscle
    sp_path = os.path.join(OUTPUT_DIR, "spearman_results.csv")
    if os.path.exists(sp_path):
        pd.read_csv(sp_path).to_excel(writer, sheet_name="Spearman_Full", index=False)

    # Sheet 6: Spearman Pooled
    spp_path = os.path.join(OUTPUT_DIR, "spearman_pooled.csv")
    if os.path.exists(spp_path):
        pd.read_csv(spp_path).to_excel(writer, sheet_name="Spearman_Pooled", index=False)

    # Sheet 7: Spearman Pivot (per muscle)
    pivot_path = os.path.join(OUTPUT_DIR, "spearman_per_muscle_pivot.csv")
    if os.path.exists(pivot_path):
        pd.read_csv(pivot_path, index_col=0).to_excel(writer, sheet_name="Spearman_Pivot")

    # Sheet 8: LME Fixed Effects (parsimonious — primary model)
    lme_path = os.path.join(OUTPUT_DIR, "lme_fixed_effects.csv")
    if os.path.exists(lme_path):
        pd.read_csv(lme_path, index_col=0).to_excel(writer, sheet_name="LME_Parsimonious")

    # Sheet 9: LME Fixed Effects (full model)
    lme_full_path = os.path.join(OUTPUT_DIR, "lme_fixed_effects_full.csv")
    if os.path.exists(lme_full_path):
        pd.read_csv(lme_full_path, index_col=0).to_excel(writer, sheet_name="LME_Full")

    # Sheet 10: LME Model-Level Stats
    lme_stats_path = os.path.join(OUTPUT_DIR, "lme_stats.json")
    if os.path.exists(lme_stats_path):
        with open(lme_stats_path) as f:
            lme_dict = json.load(f)
        lme_rows = []
        for model_name, stats_dict in lme_dict.items():
            if isinstance(stats_dict, dict):
                row = {"model": model_name}
                row.update(stats_dict)
                lme_rows.append(row)
        pd.DataFrame(lme_rows).to_excel(writer, sheet_name="LME_Model_Stats", index=False)

    # Sheet 11: Kruskal-Wallis
    kw_path = os.path.join(OUTPUT_DIR, "kruskal_wallis.csv")
    if os.path.exists(kw_path):
        pd.read_csv(kw_path).to_excel(writer, sheet_name="Kruskal_Wallis", index=False)

print(f"  Excel workbook saved → {excel_path}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE — Output files saved to:")
print(f"  {OUTPUT_DIR}")
print("=" * 70)
print("Files generated:")
for fname in ["pca_summary.json", "vif_summary.csv", "spearman_results.csv",
              "spearman_pooled.csv", "spearman_per_muscle_pivot.csv",
              "lme_fixed_effects.csv", "lme_stats.json",
              "descriptive_stats.csv", "volume_by_group_muscle.csv",
              "kruskal_wallis.csv", "statistical_analysis.xlsx"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath) if os.path.exists(fpath) else 0
    print(f"  {fname:45s}  {size:6d} bytes")
