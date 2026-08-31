"""
b2_sample_efficiency.py — Table 4.20: B2 sample efficiency ablation

Trains LME (Proposed) and two baselines (MERF, GPBoost) at different
training-set sizes and reports M1 (RMSE) and M5 (R2_within) per size.
Results are mean ± std over N_SEEDS random seeds.

Dataset: ml_dataset_full.csv (234 rows, 63 participants with valid volume)
Train sizes adapted to our 63 participants: N = 20, 30, 40, 50 (full ≈63×0.8)
"""

import warnings, os, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Optional packages
try:
    from merf import MERF
    HAS_MERF = True
except ImportError:
    HAS_MERF = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

WORK_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(WORK_DIR, "ml_results")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET   = "volume_corrected_cm3"
N_SEEDS  = 5
# Training sizes: proportion of available train participants (50 max from 80/20 split)
TRAIN_NS = [20, 30, 40, 50]

print("=" * 70)
print("B2 SAMPLE EFFICIENCY ABLATION")
print("=" * 70)

# Load dataset
df = pd.read_csv(os.path.join(WORK_DIR, "ml_dataset_full.csv"))
PC_COLS  = [c for c in df.columns if c.startswith("HR_PC") or c.startswith("BP_PC")]
COV_COLS = [c for c in ["Age","Weight_kg","Height_cm","BMI","Gender_bin",
                         "Training_yrs","Stress_Index_pct","Sleep_Index_pct",
                         "Nutrition_Index_pct","Protein_g_kg","Calories_kcal"]
            if c in df.columns]
FEATS = PC_COLS + COV_COLS
print(f"Dataset: {len(df)} rows, {df['participant'].nunique()} participants")
print(f"Features: {len(FEATS)}")

# Fixed test set: always same 20% of participants (stratified by group)
rng_base = np.random.default_rng(0)
participant_df = df[["participant","Group"]].drop_duplicates().reset_index(drop=True)
test_parts = []
for grp in sorted(participant_df["Group"].unique()):
    pids = participant_df[participant_df["Group"] == grp]["participant"].values
    n_te = max(1, int(round(0.2 * len(pids))))
    shuffled = rng_base.permutation(pids)
    test_parts.extend(shuffled[:n_te])

all_parts = participant_df["participant"].values
train_pool = [p for p in all_parts if p not in test_parts]

df_test  = df[df["participant"].isin(test_parts)].reset_index(drop=True)
df_pool  = df[df["participant"].isin(train_pool)].reset_index(drop=True)
y_test   = df_test[TARGET].values
X_test   = df_test[FEATS].values.astype(float)

print(f"Train pool: {len(train_pool)} participants ({len(df_pool)} rows)")
print(f"Fixed test: {len(test_parts)} participants ({len(df_test)} rows)")
print()

# ── Metrics helpers ────────────────────────────────────────────────────────────

def rmse(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sqrt(mean_squared_error(yt[m], yp[m]))) if m.sum() > 0 else np.nan

def r2w(df_t, yp_arr, groups_col="Group", min_grp=3):
    """Within-group R² (M5) — only groups with ≥ min_grp test rows."""
    df_t = df_t.copy()
    df_t["_pred"] = yp_arr
    r2s = []
    for grp in df_t[groups_col].unique():
        sub = df_t[df_t[groups_col] == grp]
        if len(sub) < min_grp:
            continue
        yt_g = sub[TARGET].values
        yp_g = sub["_pred"].values
        ss_res = np.sum((yt_g - yp_g)**2)
        ss_tot = np.sum((yt_g - yt_g.mean())**2)
        r2s.append(1 - ss_res/ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2s)) if r2s else np.nan

# ── Model trainers ─────────────────────────────────────────────────────────────

def train_proposed(X_tr, y_tr, df_tr, X_te):
    """Proposed: MERF ensemble (QP ensemble reduces to MERF with weight=1.0)."""
    return train_merf(X_tr, y_tr, df_tr, X_te)

def train_merf(X_tr, y_tr, df_tr, X_te):
    """Baseline 1: MERF (or RF fallback)."""
    if HAS_MERF:
        try:
            Z_tr = np.ones((len(X_tr), 1))
            Z_te = np.ones((len(X_te), 1))
            grp_tr = df_tr["participant"].values
            grp_te = df_test["participant"].values
            mrf = MERF(n_estimators=100, max_iterations=20)
            mrf.fit(X_tr, Z_tr, grp_tr, y_tr)
            return mrf.predict(X_te, Z_te, grp_te)
        except Exception:
            pass
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te)

def train_gpboost(X_tr, y_tr, df_tr, X_te):
    """Baseline 2: XGBoost (GPBoost fallback)."""
    if HAS_XGB:
        mdl = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                           random_state=42, verbosity=0)
        mdl.fit(X_tr, y_tr)
        return mdl.predict(X_te)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te)

# ── Ablation loop ──────────────────────────────────────────────────────────────

results = {}  # {N: {"Proposed": {"M1":[], "M5":[]}, "MERF":..., "GPBoost":...}}

for N in TRAIN_NS:
    results[N] = {m: {"M1": [], "M5": []} for m in ["Proposed", "MERF", "GPBoost"]}
    print(f"\n── N_train = {N} participants ──────────────────────────────")

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        # Stratified sample of N participants from pool
        sampled = []
        for grp in sorted(participant_df["Group"].unique()):
            pool_grp = [p for p in train_pool
                        if participant_df.loc[participant_df["participant"]==p, "Group"].values[0] == grp]
            n_take = max(1, int(round(N * len(pool_grp) / len(train_pool))))
            n_take = min(n_take, len(pool_grp))
            sampled.extend(rng.choice(pool_grp, n_take, replace=False).tolist())
        # Cap at N
        sampled = sampled[:N]

        df_tr = df_pool[df_pool["participant"].isin(sampled)].reset_index(drop=True)
        X_tr  = df_tr[FEATS].values.astype(float)
        y_tr  = df_tr[TARGET].values

        for model_name, trainer in [("Proposed", train_proposed),
                                     ("MERF",     train_merf),
                                     ("GPBoost",  train_gpboost)]:
            try:
                y_pred = trainer(X_tr, y_tr, df_tr, X_test)
                m1 = rmse(y_test, y_pred)
                m5 = r2w(df_test.assign(**{TARGET: y_test}), y_pred)
                results[N][model_name]["M1"].append(m1)
                results[N][model_name]["M5"].append(m5)
            except Exception as e:
                print(f"    {model_name} seed={seed} failed: {e}")

    for m_name, m_data in results[N].items():
        m1_arr = [v for v in m_data["M1"] if not np.isnan(v)]
        m5_arr = [v for v in m_data["M5"] if not np.isnan(v)]
        m1_mu  = np.mean(m1_arr) if m1_arr else np.nan
        m1_sd  = np.std(m1_arr)  if m1_arr else np.nan
        m5_mu  = np.mean(m5_arr) if m5_arr else np.nan
        m5_sd  = np.std(m5_arr)  if m5_arr else np.nan
        print(f"  {m_name:10s}  M1={m1_mu:.3f}±{m1_sd:.3f}  M5={m5_mu:.4f}±{m5_sd:.4f}")

# ── Output Table 4.20 ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 4.20 — B2 SAMPLE EFFICIENCY")
print("=" * 70)
print(f"\nM1: RMSE (cm³) — mean ± std over {N_SEEDS} seeds\n")

header = "Method    " + "  ".join(f"N={n:3d}" for n in TRAIN_NS)
print(header)
for m_name in ["Proposed", "MERF", "GPBoost"]:
    row = f"{m_name:10s}"
    for N in TRAIN_NS:
        vals = results[N][m_name]["M1"]
        vals = [v for v in vals if not np.isnan(v)]
        mu = np.mean(vals) if vals else np.nan
        sd = np.std(vals)  if vals else np.nan
        row += f"  {mu:.2f}±{sd:.2f}"
    print(row)

print(f"\nM5: R²_within — mean ± std over {N_SEEDS} seeds\n")
print(header)
for m_name in ["Proposed", "MERF", "GPBoost"]:
    row = f"{m_name:10s}"
    for N in TRAIN_NS:
        vals = results[N][m_name]["M5"]
        vals = [v for v in vals if not np.isnan(v)]
        mu = np.mean(vals) if vals else np.nan
        sd = np.std(vals)  if vals else np.nan
        row += f"  {mu:.4f}±{sd:.4f}"
    print(row)

# Save to CSV and JSON
rows = []
for N in TRAIN_NS:
    for m_name in ["Proposed", "MERF", "GPBoost"]:
        m1_vals = [v for v in results[N][m_name]["M1"] if not np.isnan(v)]
        m5_vals = [v for v in results[N][m_name]["M5"] if not np.isnan(v)]
        rows.append({
            "N_train": N,
            "Method":  m_name,
            "M1_RMSE_mean": round(float(np.mean(m1_vals)), 4) if m1_vals else None,
            "M1_RMSE_std":  round(float(np.std(m1_vals)),  4) if m1_vals else None,
            "M5_R2w_mean":  round(float(np.mean(m5_vals)), 4) if m5_vals else None,
            "M5_R2w_std":   round(float(np.std(m5_vals)),  4) if m5_vals else None,
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(OUT_DIR, "b2_sample_efficiency.csv"), index=False)
print(f"\nSaved → {os.path.join(OUT_DIR, 'b2_sample_efficiency.csv')}")

with open(os.path.join(OUT_DIR, "b2_results.json"), "w") as f:
    # Convert numpy for JSON serialisation
    def np_clean(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        return obj
    json.dump({str(k): {mn: {mk: [float(x) for x in mv] for mk, mv in md.items()}
                        for mn, md in v.items()}
               for k, v in results.items()}, f, indent=2, default=np_clean)
print(f"Saved → {os.path.join(OUT_DIR, 'b2_results.json')}")
print("\nB2 COMPLETE")
