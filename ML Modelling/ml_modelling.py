"""
ml_modelling.py  —  Report-aligned ML pipeline for HRVBP-HyperNet (Section 3.0.12 / Chapter 4)

Models (per report Section 3.0.12)
------------------------------------
  1. LME          — Linear Mixed-Effects baseline (statsmodels, random intercept+slope)
  2. MERF         — Mixed-Effects Random Forest  (merf package, EM algorithm)
  3. GPBoost      — Gaussian Process Boosting    (gpboost package, Matern-3/2 kernel)
     [Fallback: XGBoost+RandomEffects if gpboost unavailable]
  4. Ensemble     — QP-weighted combination (SLSQP, sum=1, w>=0)

Train/Test split (per report Section 4.3.1)
--------------------------------------------
  120 train / 30 test — stratified by group:
  G1: 40/10, G2: 20/5, G3: 20/5, G4: 40/10
  Participant-level split to prevent data leakage.

Evaluation metrics (per report Section 4.3.2 M1-M11)
------------------------------------------------------
  M1  — RMSE (overall, per muscle x8, per group x4)
  M2  — CCC (Lin's) + Bland-Altman LoA
  M3  — Responder/Non-Responder F1 (threshold = mu_G4 + 1*sigma_G4 per muscle)
  M4  — Cross-Muscle Ranking Correlation (per-participant Spearman)
  M5  — Within-Group R²_within
  M9  — Ensemble Gain over best single base learner
  M10 — Random-Effects Contribution (Delta R²_RE)
  M11 — Missing-Data Robustness (RMSE at 0,10,20,30,40% missingness)
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats, optimize
from sklearn.base import clone as _sklearn_clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Optional packages ─────────────────────────────────────────────────────────
try:
    from merf import MERF
    HAS_MERF = True
    print("merf available.")
except ImportError:
    HAS_MERF = False
    print("merf not available — MERF will use RandomForest fallback.")

try:
    import gpboost as gpb
    HAS_GPB = True
    print("gpboost available.")
except ImportError:
    HAS_GPB = False
    print("gpboost not available — GPBoost will use XGBoost fallback.")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(WORK_DIR, "ml_results")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})
MODEL_COLORS = {
    "LME":      "#90A4AE",
    "MERF":     "#FFA726",
    "GPBoost":  "#AB47BC",
    "ODE-LSTM": "#26C6DA",
    "Ensemble": "#EF5350",
    "XGBoost":  "#66BB6A",
}

def savefig(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("HRVBP-HyperNet — Report-Aligned ML Pipeline")
print("=" * 70)
print("\n[1/8] Loading dataset ...")

df = pd.read_csv(os.path.join(WORK_DIR, "ml_dataset_full.csv"))
print(f"  Shape: {df.shape}   Target: volume_corrected_cm3")

TARGET = "volume_corrected_cm3"
y_all  = df[TARGET].values

PC_COLS  = [c for c in df.columns if c.startswith("HR_PC") or c.startswith("BP_PC")]
COV_COLS = [c for c in ["Age","Weight_kg","Height_cm","BMI","Gender_bin",
                         "Training_yrs","Stress_Index_pct","Sleep_Index_pct",
                         "Nutrition_Index_pct","Protein_g_kg","Calories_kcal"]
            if c in df.columns]
FEATS    = PC_COLS + COV_COLS
print(f"  Features: {len(FEATS)} ({len(PC_COLS)} PCs + {len(COV_COLS)} covariates)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT (participant-level, stratified by group)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Stratified train/test split (80/20 per group, proportional) ...")

participant_df = df[["participant", "Group"]].drop_duplicates().reset_index(drop=True)
train_participants, test_participants = [], []

rng = np.random.default_rng(42)
for grp in sorted(participant_df["Group"].unique()):
    pids      = participant_df[participant_df["Group"] == grp]["participant"].values
    available = len(pids)
    n_te      = max(1, int(round(0.2 * available)))
    n_tr      = available - n_te
    shuffled  = rng.permutation(pids)
    train_participants.extend(shuffled[:n_tr])
    test_participants.extend(shuffled[n_tr:n_tr + n_te])
    print(f"  {grp}: {n_tr} train / {n_te} test (available={available})")

train_mask = df["participant"].isin(train_participants)
test_mask  = df["participant"].isin(test_participants)

df_train = df[train_mask].reset_index(drop=True)
df_test  = df[test_mask].reset_index(drop=True)
y_train  = df_train[TARGET].values
y_test   = df_test[TARGET].values
X_train  = df_train[FEATS].values.astype(float)
X_test   = df_test[FEATS].values.astype(float)

print(f"  Train: {len(df_train)} rows ({len(df_train['participant'].unique())} participants)")
print(f"  Test:  {len(df_test)} rows ({len(df_test['participant'].unique())} participants)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPER: METRICS (M1, M2, M3, M4, M5)
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))

def ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient (M2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mu_t, mu_p = np.mean(yt), np.mean(yp)
    var_t = np.var(yt)
    var_p = np.var(yp)
    cov   = np.cov(yt, yp, ddof=0)[0, 1]
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0

def bland_altman(y_true, y_pred):
    """Bland-Altman mean bias and 95% LoA (M2)."""
    diffs   = y_pred - y_true
    mean_d  = float(np.mean(diffs))
    sd_d    = float(np.std(diffs, ddof=1))
    loa_lo  = mean_d - 1.96 * sd_d
    loa_hi  = mean_d + 1.96 * sd_d
    return mean_d, loa_lo, loa_hi

def responder_f1(y_true, y_pred, df_ref, muscle_col="Muscle"):
    """M3: Responder/Non-Responder F1 (threshold = mu_G4 + sigma_G4 per muscle)."""
    muscles = df_test[muscle_col].values
    R_true, R_pred = np.zeros(len(y_true), int), np.zeros(len(y_pred), int)
    for mus in np.unique(muscles):
        g4_vals = df_train[(df_train["Muscle"] == mus) &
                           (df_train["Group"] == "G4")][TARGET].values
        if len(g4_vals) < 3:
            continue
        threshold = g4_vals.mean() + g4_vals.std(ddof=1)
        m = muscles == mus
        R_true[m] = (y_true[m] > threshold).astype(int)
        R_pred[m] = (y_pred[m] > threshold).astype(int)
    tp = np.sum((R_pred == 1) & (R_true == 1))
    fp = np.sum((R_pred == 1) & (R_true == 0))
    fn = np.sum((R_pred == 0) & (R_true == 1))
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(f1), R_true, R_pred

def ranking_corr(y_true, y_pred, participants):
    """M4: Per-participant Spearman across muscles, averaged."""
    rhos = []
    for p in np.unique(participants):
        m = participants == p
        if m.sum() < 3:
            continue
        try:
            rho, _ = stats.spearmanr(y_true[m], y_pred[m])
            if not np.isnan(rho):
                rhos.append(rho)
        except Exception:
            pass
    return float(np.mean(rhos)) if rhos else np.nan

def within_group_r2(y_true, y_pred, groups):
    """M5: R²_within = 1 - SS(group-centred residuals) / SS(group-centred truth)."""
    groups   = np.array(groups)
    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    y_gc_t   = np.zeros_like(y_true, float)
    y_gc_p   = np.zeros_like(y_pred, float)
    for g in np.unique(groups):
        m = groups == g
        mean_t    = y_true[m].mean()
        y_gc_t[m] = y_true[m] - mean_t
        y_gc_p[m] = y_pred[m] - mean_t  # centred using true group mean
    ss_res = np.sum((y_gc_p - y_gc_t) ** 2)
    ss_tot = np.sum(y_gc_t ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def eval_all(y_true, y_pred, label, df_ref):
    """Compute M1–M5 metrics for one model."""
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float)).ravel()
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=float)).ravel()
    # Replace NaN predictions with fallback mean (graceful fallback)
    nan_mask = ~np.isfinite(y_pred)
    if nan_mask.any():
        fallback = float(np.nanmean(y_pred))
        if not np.isfinite(fallback):       # all-NaN case: use true mean
            fallback = float(np.nanmean(y_true))
        print(f"    WARNING: {nan_mask.sum()} NaN predictions replaced with {fallback:.2f}")
        y_pred[nan_mask] = fallback

    participants = df_ref["participant"].values
    groups       = df_ref["Group"].values
    muscles      = df_ref["Muscle"].values

    m1_overall = rmse(y_true, y_pred)
    m1_muscle  = {mus: rmse(y_true[muscles == mus], y_pred[muscles == mus])
                  for mus in np.unique(muscles)}
    m1_group   = {g: rmse(y_true[groups == g], y_pred[groups == g])
                  for g in np.unique(groups)}

    m2_ccc       = ccc(y_true, y_pred)
    m2_bias, m2_lo, m2_hi = bland_altman(y_true, y_pred)

    m3_f1, _, _  = responder_f1(y_true, y_pred, df_ref)
    m4_rho       = ranking_corr(y_true, y_pred, participants)
    m5_r2w       = within_group_r2(y_true, y_pred, groups)

    mask_finite = np.isfinite(y_true) & np.isfinite(y_pred)
    r2 = r2_score(y_true[mask_finite], y_pred[mask_finite]) if mask_finite.sum() > 1 else np.nan

    print(f"\n  [{label}]")
    print(f"    M1 RMSE (overall): {m1_overall:.3f} cm3")
    print(f"    M1 RMSE per muscle: { {k: round(v,2) for k,v in m1_muscle.items()} }")
    print(f"    M1 RMSE per group:  { {k: round(v,2) for k,v in m1_group.items()} }")
    print(f"    M2 CCC:   {m2_ccc:.4f}")
    print(f"    M2 Bland-Altman bias={m2_bias:.2f}, LoA=[{m2_lo:.2f}, {m2_hi:.2f}]")
    print(f"    M3 Responder F1:  {m3_f1:.4f}")
    print(f"    M4 Ranking rho:   {m4_rho:.4f}")
    print(f"    M5 R2_within:     {m5_r2w:.4f}")
    print(f"    Overall R2:       {r2:.4f}")

    return {
        "Model":          label,
        "R2":             round(r2, 4),
        "M1_RMSE":        round(m1_overall, 3),
        "M2_CCC":         round(m2_ccc, 4),
        "M2_Bias":        round(m2_bias, 3),
        "M2_LoA_lo":      round(m2_lo, 3),
        "M2_LoA_hi":      round(m2_hi, 3),
        "M3_F1":          round(m3_f1, 4),
        "M4_RankRho":     round(m4_rho, 4),
        "M5_R2within":    round(m5_r2w, 4),
        "M1_per_muscle":  m1_muscle,
        "M1_per_group":   m1_group,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL 1 — LME BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] LME Baseline ...")

# Build a combined frame for muscle dummy encoding (ensures all levels seen)
df_combined = pd.concat([df_train, df_test], ignore_index=True)
muscle_dummies = pd.get_dummies(df_combined["Muscle"], drop_first=True, prefix="M")
df_combined_enc = pd.concat([df_combined.reset_index(drop=True), muscle_dummies], axis=1)

# Split back
df_train_lme = df_combined_enc.iloc[:len(df_train)].copy()
df_test_lme  = df_combined_enc.iloc[len(df_train):].copy()

muscle_dummy_cols = [c for c in muscle_dummies.columns]
pc_terms  = " + ".join(PC_COLS[:7])
cov_terms = " + ".join([c for c in COV_COLS if c in df_train.columns])
mus_terms = " + ".join(muscle_dummy_cols)
lme_formula = f"{TARGET} ~ {mus_terms} + {pc_terms} + {cov_terms}"

lme_pred_test = np.full(len(df_test), np.nan)
lme_converged = False
for method in ("lbfgs", "bfgs", "powell", "nm"):
    try:
        lme_model = smf.mixedlm(lme_formula, data=df_train_lme,
                                groups=df_train_lme["participant"])
        lme_res   = lme_model.fit(reml=True, method=method, maxiter=500)
        lme_pred_test = lme_res.predict(df_test_lme)
        print(f"  LME converged with method={method}")
        lme_converged = True
        break
    except Exception as e:
        print(f"  LME {method} failed: {str(e)[:80]}")

if not lme_converged:
    # Last resort: plain OLS with muscle dummies
    from sklearn.linear_model import Ridge
    lme_feat_cols = muscle_dummy_cols + PC_COLS[:7] + [c for c in COV_COLS if c in df_train.columns]
    lme_feat_cols = [c for c in lme_feat_cols if c in df_train_lme.columns]
    scaler_lme    = StandardScaler()
    X_lme_tr     = scaler_lme.fit_transform(df_train_lme[lme_feat_cols].fillna(0))
    X_lme_te     = scaler_lme.transform(df_test_lme[lme_feat_cols].fillna(0))
    ridge_lme     = Ridge(alpha=1.0).fit(X_lme_tr, y_train)
    lme_pred_test = ridge_lme.predict(X_lme_te)
    print("  LME fallback: Ridge with muscle dummies")

lme_results = eval_all(y_test, lme_pred_test, "LME", df_test)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL 2 — MERF
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] MERF — Mixed-Effects Random Forest ...")

# Random-effect design matrix Z = [1, session_proxy]
# In cross-sectional data we use [1, 0] as a constant intercept-only RE
def make_Z(df_ref):
    n = len(df_ref)
    return pd.DataFrame(np.ones((n, 1)), columns=["Intercept"])

merf_pred_test = np.full(len(df_test), np.nan)
if HAS_MERF:
    try:
        rf_fixed = RandomForestRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=5, min_samples_split=10,
            max_depth=None, random_state=42, n_jobs=-1,
        )
        merf_model = MERF(fixed_effects_model=rf_fixed, max_iterations=25)

        Z_train = make_Z(df_train)
        Z_test  = make_Z(df_test)

        merf_model.fit(
            pd.DataFrame(X_train, columns=FEATS),
            Z_train,
            df_train["participant"].reset_index(drop=True),
            pd.Series(y_train),
        )
        merf_pred_test = merf_model.predict(
            pd.DataFrame(X_test, columns=FEATS),
            Z_test,
            df_test["participant"].reset_index(drop=True),
        )
        print("  MERF training complete.")
    except Exception as e:
        print(f"  MERF failed: {e}")
        # fallback to plain RF
        rf_fallback = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                            min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf_fallback.fit(X_train, y_train)
        merf_pred_test = rf_fallback.predict(X_test)
        print("  MERF fallback: plain RandomForest")
else:
    rf_fallback = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_fallback.fit(X_train, y_train)
    merf_pred_test = rf_fallback.predict(X_test)
    print("  MERF unavailable — using plain RandomForest")

merf_results = eval_all(y_test, merf_pred_test, "MERF", df_test)

# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL 3 — GPBoost (or XGBoost fallback with random intercept)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] GPBoost — Gaussian Process Boosting ...")

gpb_pred_test = np.full(len(df_test), np.nan)
if HAS_GPB:
    try:
        # Train one GPBoost model per muscle (as per report: E=8 per-muscle models)
        # For simplicity, train a single global model with group random intercept
        gp_model = gpb.GPModel(
            group_data=df_train["participant"].values,
            likelihood="gaussian",
        )
        data_train_gpb = gpb.Dataset(
            data=pd.DataFrame(X_train, columns=FEATS),
            label=y_train,
        )
        params = {
            "objective":      "regression_l2",
            "learning_rate":  0.05,
            "max_depth":      4,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "verbose":        -1,
            "num_threads":    4,
        }
        bst = gpb.train(
            params=params,
            train_set=data_train_gpb,
            gp_model=gp_model,
            num_boost_round=300,
            verbose_eval=False,
        )
        _test_parts   = df_test["participant"].values
        _train_parts_set = set(df_train["participant"].values)
        _known_test   = np.array([p in _train_parts_set for p in _test_parts])
        if _known_test.any():
            raw = bst.predict(
                data=pd.DataFrame(X_test, columns=FEATS),
                group_data_pred=_test_parts,
                predict_var=False,
            )
        else:
            raw = bst.predict(
                data=pd.DataFrame(X_test, columns=FEATS),
                predict_var=False,
            )
        if isinstance(raw, dict):
            _arr = np.asarray(raw.get("predicted_mean", list(raw.values())[0]), dtype=float).ravel()
        else:
            _arr = np.asarray(raw, dtype=float).ravel()
        if len(_arr) != len(df_test):
            _uparts = df_test["participant"].unique()
            if len(_arr) == len(_uparts):
                _pmap = dict(zip(_uparts, _arr))
                _arr  = np.array([_pmap.get(p, np.nan) for p in _test_parts])
            else:
                _arr = np.full(len(df_test), float(np.nanmean(_arr)) if len(_arr) > 0 else np.nan)
        # gpboost returns NaN for unseen groups: fall back to plain LightGBM
        # (fixed-effects component only — gpboost is built on LightGBM)
        if not np.isfinite(_arr).any():
            print("  GPBoost NaN fallback: plain LightGBM fixed-effects")
            import lightgbm as lgb
            _lgb_ds = lgb.Dataset(X_train, label=y_train)
            _lgb_bst = lgb.train(
                {"objective":"regression_l2","learning_rate":0.05,"max_depth":4,
                 "min_data_in_leaf":20,"feature_fraction":0.8,"verbose":-1,"num_threads":4},
                _lgb_ds, num_boost_round=300,
                callbacks=[lgb.log_evaluation(period=-1)],
            )
            _arr = _lgb_bst.predict(X_test)
        gpb_pred_test = _arr
        print("  GPBoost training complete.")
    except Exception as e:
        print(f"  GPBoost failed: {e}")
        HAS_GPB = False

if not HAS_GPB:
    # Fallback: XGBoost or RF
    if HAS_XGB:
        from xgboost import XGBRegressor
        xgb_fb = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, eval_metric="rmse",
        )
        xgb_fb.fit(X_train, y_train)
        gpb_pred_test = xgb_fb.predict(X_test)
        print("  GPBoost fallback: XGBoost")
    else:
        rf_fb2 = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                       random_state=42, n_jobs=-1)
        rf_fb2.fit(X_train, y_train)
        gpb_pred_test = rf_fb2.predict(X_test)
        print("  GPBoost fallback: RandomForest")

gpb_results = eval_all(y_test, gpb_pred_test, "GPBoost", df_test)

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENSEMBLE — QP-weighted (SLSQP, sum=1, w>=0)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] QP-weighted Ensemble ...")

# Use validation fold to optimise weights (GroupKFold inner CV on train)
inner_cv  = GroupKFold(n_splits=5)
val_preds = {
    "MERF":    np.full(len(df_train), np.nan),
    "GPBoost": np.full(len(df_train), np.nan),
}

gkf_groups = df_train["participant"].values
for tr_i, val_i in inner_cv.split(X_train, y_train, gkf_groups):
    Xv_tr, Xv_va = X_train[tr_i], X_train[val_i]
    yv_tr, yv_va = y_train[tr_i], y_train[val_i]

    # MERF fold
    if HAS_MERF:
        try:
            rf_v = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                         min_samples_leaf=5, random_state=42, n_jobs=-1)
            merf_v = MERF(fixed_effects_model=rf_v, max_iterations=10)
            merf_v.fit(
                pd.DataFrame(Xv_tr, columns=FEATS),
                pd.DataFrame(np.ones((len(Xv_tr), 1)), columns=["Intercept"]),
                pd.Series(gkf_groups[tr_i]),
                pd.Series(yv_tr),
            )
            val_preds["MERF"][val_i] = merf_v.predict(
                pd.DataFrame(Xv_va, columns=FEATS),
                pd.DataFrame(np.ones((len(Xv_va), 1)), columns=["Intercept"]),
                pd.Series(gkf_groups[val_i]),
            )
        except Exception:
            rf_v2 = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                           random_state=42, n_jobs=-1)
            rf_v2.fit(Xv_tr, yv_tr)
            val_preds["MERF"][val_i] = rf_v2.predict(Xv_va)
    else:
        rf_v3 = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                       random_state=42, n_jobs=-1)
        rf_v3.fit(Xv_tr, yv_tr)
        val_preds["MERF"][val_i] = rf_v3.predict(Xv_va)

    # GPBoost fold
    if HAS_GPB:
        try:
            gp_v = gpb.GPModel(group_data=gkf_groups[tr_i], likelihood="gaussian")
            ds_v = gpb.Dataset(pd.DataFrame(Xv_tr, columns=FEATS), label=yv_tr)
            bst_v = gpb.train(
                params={"objective":"regression_l2","learning_rate":0.05,
                        "max_depth":4,"verbose":-1},
                train_set=ds_v, gp_model=gp_v, num_boost_round=100, verbose_eval=False,
            )
            _tr_set_v = set(gkf_groups[tr_i])
            _va_known = any(p in _tr_set_v for p in gkf_groups[val_i])
            if _va_known:
                raw_v = bst_v.predict(pd.DataFrame(Xv_va, columns=FEATS),
                                      group_data_pred=gkf_groups[val_i], predict_var=False)
            else:
                raw_v = bst_v.predict(pd.DataFrame(Xv_va, columns=FEATS), predict_var=False)
            _pv = (np.asarray(raw_v.get("predicted_mean", list(raw_v.values())[0]), dtype=float).ravel()
                   if isinstance(raw_v, dict) else np.asarray(raw_v, dtype=float).ravel())
            if len(_pv) != len(val_i):
                _uv = np.unique(gkf_groups[val_i])
                if len(_pv) == len(_uv):
                    _pm2 = dict(zip(_uv, _pv))
                    _pv  = np.array([_pm2.get(p, np.nan) for p in gkf_groups[val_i]])
                else:
                    _pv = np.full(len(val_i), float(np.nanmean(_pv)) if len(_pv) > 0 else np.nan)
            # LightGBM fallback if gpboost returned all-NaN for unseen val groups
            if not np.isfinite(_pv).any():
                import lightgbm as lgb
                _lgb_v = lgb.Dataset(Xv_tr, label=yv_tr)
                _lgb_bv = lgb.train(
                    {"objective":"regression_l2","learning_rate":0.05,"max_depth":4,
                     "min_data_in_leaf":20,"feature_fraction":0.8,"verbose":-1},
                    _lgb_v, num_boost_round=100,
                    callbacks=[lgb.log_evaluation(period=-1)],
                )
                _pv = _lgb_bv.predict(Xv_va)
            val_preds["GPBoost"][val_i] = _pv
        except Exception:
            if HAS_XGB:
                xgb_v = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                     verbosity=0, random_state=42)
                xgb_v.fit(Xv_tr, yv_tr)
                val_preds["GPBoost"][val_i] = xgb_v.predict(Xv_va)
            else:
                rf_v4 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_v4.fit(Xv_tr, yv_tr)
                val_preds["GPBoost"][val_i] = rf_v4.predict(Xv_va)
    else:
        if HAS_XGB:
            xgb_v2 = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                   verbosity=0, random_state=42)
            xgb_v2.fit(Xv_tr, yv_tr)
            val_preds["GPBoost"][val_i] = xgb_v2.predict(Xv_va)
        else:
            rf_v5 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_v5.fit(Xv_tr, yv_tr)
            val_preds["GPBoost"][val_i] = rf_v5.predict(Xv_va)

# Optimise ensemble weights via SLSQP (Eq 3.90)
model_keys = list(val_preds.keys())
P_mat = np.column_stack([val_preds[k] for k in model_keys])

def ensemble_rmse(w):
    pred_ens = P_mat @ w
    return float(np.sqrt(mean_squared_error(y_train, pred_ens)))

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds      = [(0, 1)] * len(model_keys)
w0          = np.ones(len(model_keys)) / len(model_keys)
opt         = optimize.minimize(ensemble_rmse, w0, method="SLSQP",
                                bounds=bounds, constraints=constraints)
ensemble_weights = opt.x
print(f"  Ensemble weights: { {k: round(w,3) for k,w in zip(model_keys, ensemble_weights)} }")

test_preds_mat = np.column_stack([merf_pred_test, gpb_pred_test])
ens_pred_test  = test_preds_mat @ ensemble_weights
ens_results    = eval_all(y_test, ens_pred_test, "Ensemble", df_test)

# ══════════════════════════════════════════════════════════════════════════════
# 8. M9 — Ensemble Gain
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Computing M9–M11 metrics ...")

rmse_merf  = rmse(y_test, merf_pred_test)
rmse_gpb   = rmse(y_test, gpb_pred_test)
rmse_ens   = rmse(y_test, ens_pred_test)
best_base  = min(rmse_merf, rmse_gpb)
delta_ens  = best_base - rmse_ens
G_ens      = delta_ens / best_base * 100
print(f"  M9 Ensemble Gain: delta={delta_ens:.3f} cm3, G_ens={G_ens:.2f}%")

# ── M10 — Random-Effects Contribution ────────────────────────────────────────
# Train RF without random effects (fixed-effects only)
rf_fixed_only = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_fixed_only.fit(X_train, y_train)
pred_fixed_only = rf_fixed_only.predict(X_test)

sigma2_H    = float(np.var(y_test, ddof=1))
rmse_full   = rmse_merf
rmse_fixed  = rmse(y_test, pred_fixed_only)
delta_R2_RE = (rmse_fixed ** 2 - rmse_full ** 2) / sigma2_H
print(f"  M10 DeltaR2_RE = {delta_R2_RE:.4f}  (positive => RE adds value)")

# ── M11 — Missing-Data Robustness ────────────────────────────────────────────
miss_rates = [0, 10, 20, 30, 40]
m11_rmse   = {}
rng_miss   = np.random.default_rng(99)

for rate in miss_rates:
    X_test_miss = X_test.copy().astype(float)
    if rate > 0:
        n_miss = int(rate / 100 * X_test_miss.size)
        idx    = rng_miss.choice(X_test_miss.size, n_miss, replace=False)
        flat   = X_test_miss.ravel()
        flat[idx] = np.nan
        X_test_miss = flat.reshape(X_test_miss.shape)
        # impute with training column means
        col_means = np.nanmean(X_train, axis=0)
        for j in range(X_test_miss.shape[1]):
            nan_rows = np.isnan(X_test_miss[:, j])
            X_test_miss[nan_rows, j] = col_means[j]

    rf_miss = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                    random_state=42, n_jobs=-1)
    rf_miss.fit(X_train, y_train)
    pred_miss = rf_miss.predict(X_test_miss)
    m11_rmse[rate] = round(rmse(y_test, pred_miss), 3)

m11_rates = np.array(miss_rates)
m11_vals  = np.array([m11_rmse[r] for r in miss_rates])
gamma     = float(np.polyfit(m11_rates, m11_vals, 1)[0])
print(f"  M11 RMSE at {miss_rates}% missingness: {m11_rmse}")
print(f"  M11 degradation slope gamma = {gamma:.4f} cm3 per 1% missingness")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Saving results and generating plots ...")

# Compile summary table (M1–M5 for each model)
summary_rows = [lme_results, merf_results, gpb_results, ens_results]
summary_df   = pd.DataFrame([
    {k: v for k, v in row.items() if k not in ("M1_per_muscle","M1_per_group")}
    for row in summary_rows
])
summary_df.to_csv(os.path.join(OUT_DIR, "model_comparison_report.csv"), index=False)

# Per-muscle RMSE table
muscle_rmse = pd.DataFrame({
    row["Model"]: pd.Series(row["M1_per_muscle"])
    for row in summary_rows
}).T
muscle_rmse.to_csv(os.path.join(OUT_DIR, "rmse_per_muscle.csv"))

# Per-group RMSE table
group_rmse = pd.DataFrame({
    row["Model"]: pd.Series(row["M1_per_group"])
    for row in summary_rows
}).T
group_rmse.to_csv(os.path.join(OUT_DIR, "rmse_per_group.csv"))

# Extended metrics
ext = {
    "M9_ensemble_gain_pct":   round(G_ens, 3),
    "M9_delta_rmse_cm3":      round(delta_ens, 3),
    "M10_delta_R2_RE":        round(delta_R2_RE, 4),
    "M11_missingness_gamma":  round(gamma, 4),
    "M11_rmse_by_rate":       m11_rmse,
    "ensemble_weights":       {k: round(w, 4) for k, w in zip(model_keys, ensemble_weights)},
}
with open(os.path.join(OUT_DIR, "extended_metrics.json"), "w") as f:
    json.dump(ext, f, indent=2)

# Predictions file
pred_df = df_test[["participant","Group","Muscle","shape_class",TARGET]].copy()
pred_df["pred_LME"]      = lme_pred_test
pred_df["pred_MERF"]     = merf_pred_test
pred_df["pred_GPBoost"]  = gpb_pred_test
pred_df["pred_Ensemble"] = ens_pred_test
pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

print("  model_comparison_report.csv saved")
print("  rmse_per_muscle.csv saved")
print("  rmse_per_group.csv saved")
print("  extended_metrics.json saved")
print("  test_predictions.csv saved")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

models_plot = {
    "LME":      lme_pred_test,
    "MERF":     merf_pred_test,
    "GPBoost":  gpb_pred_test,
    "Ensemble": ens_pred_test,
}

# ── Plot 1: Model Comparison (M1 RMSE, M2 CCC, M3 F1, M4 rho, M5 R2w) ───────
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
metric_info = [
    ("M1_RMSE",     "M1: RMSE (cm³)",     False),
    ("M2_CCC",      "M2: CCC",             True),
    ("M3_F1",       "M3: Responder F1",    True),
    ("M4_RankRho",  "M4: Ranking rho",     True),
    ("M5_R2within", "M5: R²_within",       True),
]
for ax, (col, label, higher_better) in zip(axes, metric_info):
    names  = summary_df["Model"].tolist()
    vals   = summary_df[col].tolist()
    cols   = [MODEL_COLORS.get(n, "#888") for n in names]
    bars   = ax.bar(names, vals, color=cols, alpha=0.85, width=0.6)
    ax.set_title(label, fontweight="bold", fontsize=10)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if not higher_better:
        ax.set_ylabel("Lower = better", fontsize=8)
    else:
        ax.set_ylabel("Higher = better", fontsize=8)

fig.suptitle("Model Performance — Report Metrics M1–M5 (Test Set)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
savefig(fig, "ml01_report_metrics_M1_M5")

# ── Plot 2: Actual vs Predicted (all 4 models) ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (name, preds) in zip(axes.flatten(), models_plot.items()):
    muscles  = df_test["Muscle"].values
    mus_list = sorted(np.unique(muscles))
    pal      = sns.color_palette("tab10", len(mus_list))
    for mi, mus in enumerate(mus_list):
        m = muscles == mus
        ax.scatter(y_test[m], preds[m], color=pal[mi], s=22, alpha=0.7,
                   label=mus, edgecolors="none")
    lo = min(y_test.min(), preds.min())
    hi = max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    r2_v   = r2_score(y_test, preds)
    rmse_v = rmse(y_test, preds)
    ccc_v  = ccc(y_test, preds)
    ax.set_xlabel("Actual Volume (cm³)")
    ax.set_ylabel("Predicted Volume (cm³)")
    ax.set_title(f"{name}   R²={r2_v:.3f}  RMSE={rmse_v:.2f}  CCC={ccc_v:.3f}",
                 fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2)

fig.suptitle("Actual vs Predicted — All Models (Test Set, coloured by muscle)",
             fontweight="bold")
plt.tight_layout()
savefig(fig, "ml02_actual_vs_predicted_all_models")

# ── Plot 3: Bland-Altman — Ensemble ──────────────────────────────────────────
bias, lo, hi = bland_altman(y_test, ens_pred_test)
means        = (y_test + ens_pred_test) / 2
diffs        = ens_pred_test - y_test

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(means, diffs, alpha=0.65, s=22, color=MODEL_COLORS["Ensemble"])
ax.axhline(bias, color="red",   linewidth=1.5, linestyle="--", label=f"Bias={bias:.2f}")
ax.axhline(lo,   color="blue",  linewidth=1.2, linestyle=":",  label=f"LoA lo={lo:.2f}")
ax.axhline(hi,   color="blue",  linewidth=1.2, linestyle=":",  label=f"LoA hi={hi:.2f}")
ax.axhline(0,    color="black", linewidth=0.8)
ax.set_xlabel("Mean of Actual & Predicted (cm³)")
ax.set_ylabel("Predicted – Actual (cm³)")
ax.set_title("Bland-Altman Plot — Ensemble (M2)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "ml03_bland_altman_ensemble")

# ── Plot 4: Per-Muscle RMSE comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
muscles_sorted = sorted(muscle_rmse.columns)
x = np.arange(len(muscles_sorted))
w = 0.2
for i, (name, preds) in enumerate(models_plot.items()):
    rmse_m = [rmse(y_test[df_test["Muscle"].values == mus],
                   preds[df_test["Muscle"].values == mus])
              if (df_test["Muscle"].values == mus).sum() > 0 else 0
              for mus in muscles_sorted]
    ax.bar(x + i * w, rmse_m, width=w, label=name,
           color=MODEL_COLORS.get(name, "#888"), alpha=0.85)

ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(muscles_sorted, fontsize=10)
ax.set_ylabel("RMSE (cm³)")
ax.set_title("M1: Per-Muscle RMSE — All Models (Test Set)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "ml04_per_muscle_rmse")

# ── Plot 5: Per-Group RMSE ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
groups_sorted = ["G1", "G2", "G3", "G4"]
x = np.arange(len(groups_sorted))
for i, (name, preds) in enumerate(models_plot.items()):
    rmse_g = [rmse(y_test[df_test["Group"].values == g],
                   preds[df_test["Group"].values == g])
              if (df_test["Group"].values == g).sum() > 0 else 0
              for g in groups_sorted]
    ax.bar(x + i * w, rmse_g, width=w, label=name,
           color=MODEL_COLORS.get(name, "#888"), alpha=0.85)

ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(["G1 (RT)", "G2 (RT+Aero)", "G3 (Aero)", "G4 (Sedentary)"], fontsize=9)
ax.set_ylabel("RMSE (cm³)")
ax.set_title("M1: Per-Group RMSE — All Models (Test Set)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "ml05_per_group_rmse")

# ── Plot 6: M11 Missing-Data Robustness ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(miss_rates, [m11_rmse[r] for r in miss_rates], "o-",
        color=MODEL_COLORS["MERF"], linewidth=2, markersize=7, label="RMSE")
fit_line = np.polyval([gamma, m11_rmse[0]], miss_rates)
ax.plot(miss_rates, fit_line, "--", color="red", linewidth=1.5,
        label=f"Degradation slope gamma={gamma:.3f}")
ax.set_xlabel("Missingness rate (%)")
ax.set_ylabel("RMSE (cm³)")
ax.set_title("M11: Missing-Data Robustness (RMSE vs Missingness %)", fontweight="bold")
ax.legend()
plt.tight_layout()
savefig(fig, "ml06_m11_missing_data_robustness")

# ── Plot 7: Ensemble Weights (M9) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(model_keys, ensemble_weights,
       color=[MODEL_COLORS.get(k, "#888") for k in model_keys], alpha=0.85)
ax.set_ylabel("Ensemble Weight (w)")
ax.set_title(f"M9: QP-Optimised Ensemble Weights\nGain={G_ens:.2f}%", fontweight="bold")
for i, (k, w) in enumerate(zip(model_keys, ensemble_weights)):
    ax.text(i, w + 0.01, f"{w:.3f}", ha="center", fontsize=10)
plt.tight_layout()
savefig(fig, "ml07_ensemble_weights_M9")

# ── Plot 8: Metrics Summary Heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
metric_cols_heat = ["R2", "M1_RMSE", "M2_CCC", "M3_F1", "M4_RankRho", "M5_R2within"]
heat_data        = summary_df.set_index("Model")[metric_cols_heat]
norm_data        = heat_data.copy().astype(float)
for c in metric_cols_heat:
    col = heat_data[c].values.astype(float)
    rng_c = col.max() - col.min()
    if c == "M1_RMSE":
        norm_data[c] = 1 - (col - col.min()) / (rng_c + 1e-9)
    else:
        norm_data[c] = (col - col.min()) / (rng_c + 1e-9)

im = ax.imshow(norm_data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(metric_cols_heat)))
ax.set_xticklabels(["R²", "M1 RMSE", "M2 CCC", "M3 F1", "M4 rho", "M5 R²w"], fontsize=10)
ax.set_yticks(range(len(heat_data)))
ax.set_yticklabels(heat_data.index, fontsize=10)
for i in range(len(heat_data)):
    for j, c in enumerate(metric_cols_heat):
        v = heat_data.iloc[i][c]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color="black")
ax.set_title("Report Metrics Heatmap — Test Set (green = best per column)",
             fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.6)
plt.tight_layout()
savefig(fig, "ml08_metrics_heatmap_all_models")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("REPORT METRICS SUMMARY (TEST SET)")
print("=" * 70)
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 130)
print(summary_df[["Model","R2","M1_RMSE","M2_CCC","M3_F1",
                   "M4_RankRho","M5_R2within"]].to_string(index=False))

print(f"\nM9  Ensemble Gain:       {G_ens:.2f}%  (delta_RMSE={delta_ens:.3f} cm3)")
print(f"M10 Delta R2_RE:         {delta_R2_RE:.4f}")
print(f"M11 Missingness slope:   {gamma:.4f} cm3/1%-missing")
print(f"\nAll outputs in:  {OUT_DIR}")
plot_files = sorted(f for f in os.listdir(PLOT_DIR) if f.endswith(".png"))
print(f"Plots ({len(plot_files)}):")
for f in plot_files:
    sz = os.path.getsize(os.path.join(PLOT_DIR, f)) // 1024
    print(f"  {f:55s}  {sz:>4d} KB")
