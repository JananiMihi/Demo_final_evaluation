"""
generate_plots.py  —  Complete visualisation suite for the HRVBP-HyperNet study.
Run after statistical_analysis.py has generated output files.
Produces 12 publication-ready figures in plots/ subdirectory.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET = "volume_corrected_cm3"
MUSCLE_ORDER = ["BB", "BF", "DL", "FDS", "GA", "TA", "TB", "VL"]
SHAPE_MAP = {
    "BB": "Fusiform", "TB": "Fusiform",
    "DL": "Pennate",  "GA": "Pennate",
    "FDS": "Unipennate", "TA": "Unipennate",
    "BF": "Strap",    "VL": "Strap",
}
SHAPE_COLORS = {
    "Fusiform": "#E91E63", "Pennate": "#2196F3",
    "Unipennate": "#4CAF50", "Strap": "#FF9800",
}
GROUP_COLORS  = {"G1": "#1565C0", "G2": "#2E7D32", "G3": "#E65100", "G4": "#7B1FA2"}
GROUP_LABELS  = {
    "G1": "G1: Pure RT", "G2": "G2: RT+Aerobic",
    "G3": "G3: Aerobic", "G4": "G4: Sedentary",
}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

def save(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}.png")

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════════════
print("Loading output files...")
spearman_df  = pd.read_csv(os.path.join(WORK_DIR, "spearman_results.csv"),  index_col=0)
spearman_piv = pd.read_csv(os.path.join(WORK_DIR, "spearman_per_muscle_pivot.csv"), index_col=0)
vif_df       = pd.read_csv(os.path.join(WORK_DIR, "vif_summary.csv"), index_col=0)
lme_fe       = pd.read_csv(os.path.join(WORK_DIR, "lme_fixed_effects.csv"), index_col=0)
kw_df        = pd.read_csv(os.path.join(WORK_DIR, "kruskal_wallis.csv"), index_col=0)
vol_gm_raw   = pd.read_csv(os.path.join(WORK_DIR, "volume_by_group_muscle.csv"))
with open(os.path.join(WORK_DIR, "pca_summary.json"))  as f: pca_summary = json.load(f)
with open(os.path.join(WORK_DIR, "lme_stats.json"))    as f: lme_stats   = json.load(f)

spearman_df = spearman_df.reset_index()   # make Component a regular column
sig_pairs   = spearman_df[spearman_df["Significant"] == True].copy()

# ══════════════════════════════════════════════════════════════════════════════
# RE-RUN PREPROCESSING + PCA  (for diagnostic plots / scatter plots)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading preprocessed dataset...")

PREPROCESSED_PATH = os.path.join(WORK_DIR, "Full_Data_Set_preprocessed.xlsx")
df_raw  = pd.read_excel(PREPROCESSED_PATH, sheet_name="preprocessed")
df      = df_raw.copy()
df["Group"]  = df["Group"].astype(str).str.strip()
df["Muscle"] = df["Muscle"].astype(str).str.strip()
df = df[df["Muscle"].notna() & (df["Muscle"] != "TR")].reset_index(drop=True)

df_main = df[df[TARGET].notna()].reset_index(drop=True)
if "Gender_bin" not in df_main.columns:
    df_main["Gender_bin"] = (df_main["Gender"].astype(str).str.strip().str.lower() == "male").astype(int)
if "Group_code" not in df_main.columns:
    df_main["Group_code"] = df_main["Group"].map({"G1":1,"G2":2,"G3":3,"G4":4}).fillna(0).astype(int)

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

all_feat_cols = HR_FEATURES + BP_FEATURES + COVARIATE_COLS + [TARGET]
existing_feat = [c for c in all_feat_cols if c in df_main.columns]
for col in existing_feat:
    df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
for col in existing_feat:
    if df_main[col].isna().any():
        try:
            medians = df_main.groupby("Muscle")[col].transform("median")
            df_main[col] = df_main[col].fillna(medians)
        except Exception:
            pass
        gmed = df_main[col].median()
        if pd.notna(gmed):
            df_main[col] = df_main[col].fillna(gmed)

LOG_CANDIDATES = [c for c in HR_FEATURES + BP_FEATURES
                  if any(kw in c for kw in ["abs","power","Total_power","RMSSD","SDNN","TINN","SD1","SD2","SampEn"])]
for col in LOG_CANDIDATES:
    if col in df_main.columns and (df_main[col] > 0).all():
        df_main[col + "_log"] = np.log1p(df_main[col])

def use_log(feat_list, df):
    out = []
    for c in feat_list:
        out.append(c + "_log" if (c + "_log") in df.columns else c)
    return list(dict.fromkeys(out))

HR_FEAT = [c for c in use_log(HR_FEATURES, df_main) if c in df_main.columns]
BP_FEAT = [c for c in use_log(BP_FEATURES, df_main) if c in df_main.columns]

scaler_hr = StandardScaler()
scaler_bp = StandardScaler()
X_hr_scaled = scaler_hr.fit_transform(df_main[HR_FEAT].values.astype(float))
X_bp_scaled = scaler_bp.fit_transform(df_main[BP_FEAT].values.astype(float))

def fit_pca(X_scaled, variance_threshold=0.95):
    pca_full = PCA().fit(X_scaled)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp   = int(np.argmax(cumvar >= variance_threshold) + 1)
    pca      = PCA(n_components=n_comp).fit(X_scaled)
    scores   = pca.transform(X_scaled)
    return pca, scores, n_comp, pca_full.explained_variance_ratio_, cumvar

pca_hr, scores_hr, n_hr, ev_hr_full, cumvar_hr = fit_pca(X_hr_scaled)
pca_bp, scores_bp, n_bp, ev_bp_full, cumvar_bp = fit_pca(X_bp_scaled)

pc_labels = [f"HR_PC{i+1}" for i in range(n_hr)] + [f"BP_PC{i+1}" for i in range(n_bp)]
Z = np.hstack([scores_hr, scores_bp])
df_pca = pd.DataFrame(Z, columns=pc_labels)
df_pca["participant"] = df_main["participant"].values
df_pca["Muscle"]      = df_main["Muscle"].values
df_pca["Group"]       = df_main["Group"].values
df_pca[TARGET]        = df_main[TARGET].values
for cov in COVARIATE_COLS:
    if cov in df_main.columns:
        df_pca[cov] = df_main[cov].values

print(f"  Preprocessing done. N={len(df_main)}, HR_PCs={n_hr}, BP_PCs={n_bp}")

# ══════════════════════════════════════════════════════════════════════════════
# Re-fit LME (parsimonious) for residual diagnostics
# ══════════════════════════════════════════════════════════════════════════════
print("  Re-fitting parsimonious LME for residual diagnostics...")
df_lme = df_pca.copy()
y_mean, y_std = df_lme[TARGET].mean(), df_lme[TARGET].std()
df_lme["volume_z"] = (df_lme[TARGET] - y_mean) / y_std

spearman_sig_pcs = list(sig_pairs["Component"].unique()) if len(sig_pairs) > 0 else pc_labels[:5]
key_covs = [c for c in ["Age","BMI","Gender_bin"] if c in df_lme.columns]

def _sanitise(c):
    return c.replace(" ","_").replace("(","").replace(")","").replace("/","_").replace("-","_").replace("%","pct")

df_lme.columns = [_sanitise(c) for c in df_lme.columns]
spearman_sig_pcs = [_sanitise(c) for c in spearman_sig_pcs]
key_covs         = [_sanitise(c) for c in key_covs]

TARGET_Z = "volume_z"
fe_cols  = [c for c in spearman_sig_pcs + key_covs if c in df_lme.columns]
df_fit   = df_lme.dropna(subset=[TARGET_Z, "Muscle", "participant"] + fe_cols).copy()
fe_cols  = [c for c in fe_cols if df_fit[c].std() > 1e-10]
formula  = TARGET_Z + " ~ C(Muscle) + " + " + ".join(fe_cols)

lme_result = None
for method in ("bfgs","lbfgs","gradient","powell"):
    try:
        mdl  = smf.mixedlm(formula, data=df_fit, groups=df_fit["participant"])
        lme_result = mdl.fit(reml=True, method=method, maxiter=500)
        print(f"  LME refitted via {method}.")
        break
    except Exception:
        pass

if lme_result is None:
    print("  LME refit failed — residual plots will be skipped.")

print("Done. Generating figures...")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: MUSCLE VOLUME BY TRAINING GROUP  (violin + strip)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/12] Volume by Group")
fig, ax = plt.subplots(figsize=(9, 5))
groups = ["G1","G2","G3","G4"]
group_data = [df_main.loc[df_main["Group"]==g, TARGET].dropna().values for g in groups]

parts = ax.violinplot(group_data, positions=range(len(groups)),
                      widths=0.7, showmedians=True, showextrema=True)
for i, (body, g) in enumerate(zip(parts["bodies"], groups)):
    body.set_facecolor(GROUP_COLORS[g])
    body.set_alpha(0.6)
    body.set_edgecolor("grey")
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color("grey")
parts["cmins"].set_color("grey")
parts["cmaxes"].set_color("grey")

rng = np.random.default_rng(42)
for i, (d, g) in enumerate(zip(group_data, groups)):
    jitter = rng.uniform(-0.12, 0.12, len(d))
    ax.scatter(i + jitter, d, alpha=0.45, s=14,
               color=GROUP_COLORS[g], zorder=3, edgecolors="none")
    ax.text(i, np.median(d) + 6, f"Md={np.median(d):.1f}", ha="center",
            fontsize=7.5, color="black", fontweight="bold")

ax.set_xticks(range(len(groups)))
ax.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=9)
ax.set_ylabel("Muscle Volume (cm³)")
ax.set_title("Skeletal Muscle Volume Distribution by Training Group", fontweight="bold")
ax.set_xlabel("")

handles = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g], alpha=0.7)
           for g in groups]
ax.legend(handles=handles, loc="upper right", frameon=True)
save(fig, "fig01_volume_by_group")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: VOLUME BY MUSCLE TYPE  (box, colored by shape class)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/12] Volume by Muscle")
muscles = [m for m in MUSCLE_ORDER if m in df_main["Muscle"].unique()]
fig, ax = plt.subplots(figsize=(10, 5))

muscle_data = [df_main.loc[df_main["Muscle"]==m, TARGET].dropna().values for m in muscles]
bp = ax.boxplot(muscle_data, positions=range(len(muscles)), widths=0.55,
                patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))

for i, (patch, m) in enumerate(zip(bp["boxes"], muscles)):
    shape = SHAPE_MAP.get(m, "Unknown")
    patch.set_facecolor(SHAPE_COLORS.get(shape, "#888"))
    patch.set_alpha(0.7)
    n = len(muscle_data[i])
    ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else -8, f"n={n}",
            ha="center", fontsize=7.5, color="#555")

ax.set_xticks(range(len(muscles)))
ax.set_xticklabels(muscles, fontsize=10)
ax.set_ylabel("Muscle Volume (cm³)")
ax.set_title("Skeletal Muscle Volume by Muscle Group (color = shape class)", fontweight="bold")
shape_legend = [mpatches.Patch(color=SHAPE_COLORS[s], label=s, alpha=0.7)
                for s in SHAPE_COLORS]
ax.legend(handles=shape_legend, title="Muscle Shape", loc="upper right")
save(fig, "fig02_volume_by_muscle")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: GROUP × MUSCLE MEAN VOLUME HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("[3/12] Group × Muscle Heatmap")
pivot_mean = df_main.groupby(["Group","Muscle"])[TARGET].mean().unstack("Muscle")
pivot_mean = pivot_mean.reindex(columns=[m for m in MUSCLE_ORDER if m in pivot_mean.columns])
pivot_mean = pivot_mean.reindex(["G1","G2","G3","G4"])

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(pivot_mean.values, aspect="auto", cmap="YlOrRd")
plt.colorbar(im, ax=ax, label="Mean Volume (cm³)", shrink=0.8)
ax.set_xticks(range(pivot_mean.shape[1]))
ax.set_xticklabels(pivot_mean.columns, fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels([GROUP_LABELS.get(g, g) for g in pivot_mean.index], fontsize=9)
ax.set_title("Mean Skeletal Muscle Volume (cm³) by Group × Muscle", fontweight="bold")
for i in range(pivot_mean.shape[0]):
    for j in range(pivot_mean.shape[1]):
        val = pivot_mean.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8.5, color="black" if val < 90 else "white")
save(fig, "fig03_heatmap_group_muscle")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: PCA SCREE PLOTS  (HR and BP side by side)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/12] PCA Scree Plots")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, ev_full, cumvar_arr, n_comp, domain, color in [
    (axes[0], ev_hr_full, cumvar_hr, n_hr, "HR (Cardiovascular)", "#1565C0"),
    (axes[1], ev_bp_full, cumvar_bp, n_bp, "BP (Blood Pressure)",  "#E65100"),
]:
    idx = np.arange(1, len(ev_full)+1)
    ax2 = ax.twinx()
    ax.bar(idx, ev_full*100, color=color, alpha=0.55, label="Individual %var")
    ax2.plot(idx, cumvar_arr*100, "o-", color="black", linewidth=1.5, markersize=4,
             label="Cumulative %var")
    ax2.axhline(95, color="red", linestyle="--", linewidth=1, label="95% threshold")
    ax2.axvline(n_comp, color="green", linestyle=":", linewidth=1.5,
                label=f"Retained: {n_comp} PCs")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)", color=color)
    ax2.set_ylabel("Cumulative Variance (%)", color="black")
    ax2.set_ylim(0, 105)
    ax.set_xlim(0.5, min(len(ev_full)+0.5, 20))
    ax.set_title(f"{domain} Domain PCA", fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7.5, loc="center right")
    ax.tick_params(axis="y", labelcolor=color)

fig.suptitle("Domain-wise PCA: Variance Explained", fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "fig04_pca_scree")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: PCA LOADING HEATMAP  (Spearman-significant PCs only)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/12] PCA Loading Heatmap")
sig_pc_names = list(sig_pairs["Component"].unique())

hr_sig = [pc for pc in sig_pc_names if pc.startswith("HR_")]
bp_sig = [pc for pc in sig_pc_names if pc.startswith("BP_")]

def get_loadings_df(pca_obj, feat_names, pc_prefix, pc_list):
    comp_df = pd.DataFrame(pca_obj.components_.T, index=feat_names,
                           columns=[f"{pc_prefix}_PC{i+1}" for i in range(pca_obj.n_components_)])
    return comp_df[[c for c in pc_list if c in comp_df.columns]]

hr_load = get_loadings_df(pca_hr, HR_FEAT, "HR", hr_sig)
bp_load = get_loadings_df(pca_bp, BP_FEAT, "BP", bp_sig)

def trim_loadings(load_df, threshold=0.25):
    mask = (load_df.abs() >= threshold).any(axis=1)
    return load_df[mask]

hr_load_trim = trim_loadings(hr_load, 0.28)
bp_load_trim = trim_loadings(bp_load, 0.28)

short_names = {n: n.replace("_log","").replace("GLOBAL_","G.").replace("REST_","R.")
                  .replace("EX_","E.").replace("_FFT","").replace("_ms","")
                  .replace("_pct","").replace("_mmHg","").replace("_bpm","")
               for n in list(hr_load_trim.index) + list(bp_load_trim.index)}

n_rows = max(len(hr_load_trim), len(bp_load_trim))
fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_rows * 0.35 + 1.5)))

for ax, load_df, title in [
    (axes[0], hr_load_trim, "HR Domain Loadings"),
    (axes[1], bp_load_trim, "BP Domain Loadings"),
]:
    if load_df.empty:
        ax.set_visible(False)
        continue
    data   = load_df.values
    ynames = [short_names.get(n, n) for n in load_df.index]
    xnames = list(load_df.columns)
    im     = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(xnames)))
    ax.set_xticklabels(xnames, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ynames)))
    ax.set_yticklabels(ynames, fontsize=7.5)
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            v = data[ii, jj]
            if abs(v) >= 0.28:
                ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(v) > 0.55 else "black")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Loading")

fig.suptitle("PCA Loadings for Spearman-Significant Components\n(|loading| >= 0.28 shown)",
             fontweight="bold")
plt.tight_layout()
save(fig, "fig05_pca_loadings_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: VIF BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print("[6/12] VIF Bar Chart")
vif_data = vif_df.copy()
vif_data.index = vif_data.index.astype(str)
vif_vals = vif_data["VIF"].sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, max(5, len(vif_vals)*0.3 + 1)))
colors_vif = ["#F44336" if v > 10 else "#4CAF50" for v in vif_vals]
bars = ax.barh(range(len(vif_vals)), vif_vals.values, color=colors_vif, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(vif_vals)))
ax.set_yticklabels(vif_vals.index, fontsize=7.5)
ax.axvline(10, color="red", linestyle="--", linewidth=1.5, label="VIF = 10 (threshold)")
ax.set_xlabel("Variance Inflation Factor (VIF)")
ax.set_title("VIF Screening of PC + Covariate Predictors", fontweight="bold")
ax.legend()
for i, v in enumerate(vif_vals.values):
    ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=7)
save(fig, "fig06_vif_barchart")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: SPEARMAN CORRELATION HEATMAP  (component × muscle)
# ══════════════════════════════════════════════════════════════════════════════
print("[7/12] Spearman Heatmap")
pivot = spearman_piv.copy()
# Keep only columns that exist in MUSCLE_ORDER
muscle_cols = [m for m in MUSCLE_ORDER if m in pivot.columns]
pivot       = pivot[muscle_cols]

# Convert to float
for c in pivot.columns:
    pivot[c] = pd.to_numeric(pivot[c], errors="coerce")

fig, ax = plt.subplots(figsize=(9, max(4, len(pivot)*0.7 + 1.5)))
mask = pivot.isna()
vext = max(abs(pivot.fillna(0).values.max()), abs(pivot.fillna(0).values.min()), 0.5)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
im   = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vext, vmax=vext)
plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.7)
ax.set_xticks(range(len(muscle_cols)))
ax.set_xticklabels(muscle_cols, fontsize=10)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("Spearman Correlations: PC Components vs Muscle Volume\n(|rho| >= 0.30, p < 0.05)",
             fontweight="bold")

for ii in range(pivot.shape[0]):
    for jj in range(pivot.shape[1]):
        val = pivot.values[ii, jj]
        if not np.isnan(val) and abs(val) >= 0.30:
            ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if abs(val) > 0.6 else "black",
                    fontweight="bold")
save(fig, "fig07_spearman_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: SPEARMAN DOT PLOT (forest-style, significant pairs)
# ══════════════════════════════════════════════════════════════════════════════
print("[8/12] Spearman Dot Plot")
if len(sig_pairs) > 0:
    sp_sorted = sig_pairs.sort_values("rho", ascending=True).copy()
    sp_sorted["label"] = sp_sorted["Component"] + " – " + sp_sorted["Muscle"]
    ci_half = 1.96 / np.sqrt(sp_sorted["n"].values - 3)   # Fisher z approx SE

    fig, ax = plt.subplots(figsize=(9, max(4, len(sp_sorted)*0.55 + 1.5)))
    y_pos = range(len(sp_sorted))
    colors_dot = ["#E65100" if r < 0 else "#1565C0" for r in sp_sorted["rho"]]
    ax.barh(list(y_pos), sp_sorted["rho"].values, color=colors_dot, alpha=0.7, height=0.6)
    ax.errorbar(sp_sorted["rho"].values, list(y_pos), xerr=ci_half,
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sp_sorted["label"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.30, color="green", linestyle="--", linewidth=1, label="|rho|=0.30 threshold")
    ax.axvline(-0.30, color="green", linestyle="--", linewidth=1)
    for i, (_, row) in enumerate(sp_sorted.iterrows()):
        stars = sig_stars(row["p_value"])
        ax.text(row["rho"] + (0.02 if row["rho"] >= 0 else -0.04), i,
                stars, va="center", ha="left" if row["rho"] >= 0 else "right",
                fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Spearman rho (± approximate 95% CI)")
    ax.set_title("Significant Spearman Correlations: PC Scores vs Muscle Volume\n(p < 0.05, |rho| >= 0.30)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, "fig08_spearman_dotplot")
else:
    print("  No significant pairs — skipped.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: SCATTER PLOTS — top significant Spearman pairs
# ══════════════════════════════════════════════════════════════════════════════
print("[9/12] Scatter Plots (Spearman pairs)")
if len(sig_pairs) > 0:
    top_pairs = sig_pairs.sort_values("rho", key=abs, ascending=False).head(6)
    ncols = min(3, len(top_pairs))
    nrows = int(np.ceil(len(top_pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5))
    axes_flat = np.array(axes).flatten()

    for ax_i, (_, row) in enumerate(top_pairs.iterrows()):
        ax  = axes_flat[ax_i]
        comp = row["Component"]
        musc = row["Muscle"]
        mask = df_pca["Muscle"] == musc
        x    = df_pca.loc[mask, comp].values
        y    = df_pca.loc[mask, TARGET].values
        grp  = df_pca.loc[mask, "Group"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y, grp = x[valid], y[valid], grp[valid]

        for g in np.unique(grp):
            gm = grp == g
            ax.scatter(x[gm], y[gm], color=GROUP_COLORS.get(g, "#888"),
                       s=30, alpha=0.7, label=GROUP_LABELS.get(g, g), zorder=3)

        if len(x) >= 3:
            m, b, r, pv, _ = stats.linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m*xs + b, "k-", linewidth=1.5, zorder=4)
            ax.set_title(f"{comp} vs {musc}\nrho={row['rho']:.3f}, p={row['p_value']:.3g}",
                         fontsize=9, fontweight="bold")
        ax.set_xlabel(comp, fontsize=8.5)
        ax.set_ylabel("Volume (cm³)", fontsize=8.5)

        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for ax_i in range(len(top_pairs), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle("Scatter Plots: Significant Spearman Pairs\n(black line = OLS fit within muscle)",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, "fig09_scatter_spearman_pairs")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: LME FOREST PLOT  (fixed effects ± 95% CI)
# ══════════════════════════════════════════════════════════════════════════════
print("[10/12] LME Forest Plot")
lme_clean = lme_fe.copy()
lme_clean = lme_clean.dropna(subset=["Estimate","CI_low","CI_high"])

# Separate muscle contrasts from predictor effects
is_muscle = lme_clean.index.str.startswith("C(Muscle)")
muscle_fe = lme_clean[is_muscle].copy()
other_fe  = lme_clean[~is_muscle].copy()
# Nicer labels
muscle_fe.index = muscle_fe.index.str.replace(r"C\(Muscle\)\[T\.", "", regex=True).str.replace("]","")
other_fe.index  = other_fe.index.str.replace("_", " ")

fe_all = pd.concat([other_fe, muscle_fe])
is_sig = fe_all["p_value"] < 0.05
colors_fe = ["#C62828" if s else "#90A4AE" for s in is_sig]

fig, ax = plt.subplots(figsize=(9, max(5, len(fe_all)*0.5 + 1.5)))
y_pos = range(len(fe_all))
ax.barh(list(y_pos), fe_all["Estimate"].values, color=colors_fe, alpha=0.8, height=0.6)
ax.errorbar(fe_all["Estimate"].values, list(y_pos),
            xerr=[fe_all["Estimate"] - fe_all["CI_low"], fe_all["CI_high"] - fe_all["Estimate"]],
            fmt="none", color="black", capsize=4, linewidth=1.2)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(fe_all.index, fontsize=8.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
for i, (_, row) in enumerate(fe_all.iterrows()):
    if row["p_value"] < 0.05:
        ax.text(row["Estimate"] + (0.015 if row["Estimate"] >= 0 else -0.015),
                i, sig_stars(row["p_value"]), va="center",
                ha="left" if row["Estimate"] >= 0 else "right",
                fontsize=9, fontweight="bold", color="#C62828")

sig_patch = mpatches.Patch(color="#C62828", alpha=0.8, label="p < 0.05")
ns_patch  = mpatches.Patch(color="#90A4AE", alpha=0.8, label="p >= 0.05")
ax.legend(handles=[sig_patch, ns_patch], fontsize=8.5, loc="lower right")
ax.set_xlabel("Standardised Coefficient (95% CI)")
ax.set_title("LME Fixed Effects — Parsimonious Model\n(outcome: standardised muscle volume)",
             fontweight="bold")
save(fig, "fig10_lme_forest_plot")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: LME DIAGNOSTICS  (residuals vs fitted + Q-Q)
# ══════════════════════════════════════════════════════════════════════════════
print("[11/12] LME Diagnostics")
if lme_result is not None:
    fitted_vals = lme_result.fittedvalues.values
    resid_vals  = lme_result.resid.values

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 11a: Residuals vs Fitted
    ax = axes[0]
    ax.scatter(fitted_vals, resid_vals, alpha=0.45, s=14, color="#1565C0")
    ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Fitted Values (standardised)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted", fontweight="bold")

    # 11b: Q-Q plot
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(resid_vals, dist="norm")
    ax.scatter(osm, osr, alpha=0.45, s=14, color="#E65100")
    qq_line = np.array([min(osm), max(osm)])
    ax.plot(qq_line, slope*qq_line + intercept, "r-", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Normal Q-Q Plot of Residuals", fontweight="bold")

    # 11c: Residual histogram
    ax = axes[2]
    ax.hist(resid_vals, bins=25, color="#4CAF50", alpha=0.7, edgecolor="white")
    xs = np.linspace(resid_vals.min(), resid_vals.max(), 200)
    pdf = stats.norm.pdf(xs, resid_vals.mean(), resid_vals.std()) * len(resid_vals) * \
          (resid_vals.max()-resid_vals.min()) / 25
    ax.plot(xs, pdf, "r-", linewidth=1.5, label="Normal fit")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.legend()

    shapiro_stat, shapiro_p = stats.shapiro(resid_vals)
    fig.suptitle(f"LME Model Diagnostics (Shapiro-Wilk W={shapiro_stat:.3f}, p={shapiro_p:.4f})",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, "fig11_lme_diagnostics")
else:
    print("  Skipped (LME not fitted).")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: KRUSKAL-WALLIS  (BB and VL — significant muscles)
# ══════════════════════════════════════════════════════════════════════════════
print("[12/12] Kruskal-Wallis Box Plots")
sig_muscles = kw_df[kw_df["Significant"] == True].index.tolist()
if not sig_muscles:
    sig_muscles = kw_df["H_stat"].nlargest(2).index.tolist()

ncols = len(sig_muscles)
fig, axes = plt.subplots(1, ncols, figsize=(ncols*5.5, 5))
if ncols == 1:
    axes = [axes]

for ax, muscle in zip(axes, sig_muscles):
    mask = df_main["Muscle"] == muscle
    df_m = df_main[mask].copy()
    groups_present = sorted(df_m["Group"].unique())

    data_g = [df_m.loc[df_m["Group"]==g, TARGET].dropna().values for g in groups_present]
    bp_kw  = ax.boxplot(data_g, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, g in zip(bp_kw["boxes"], groups_present):
        patch.set_facecolor(GROUP_COLORS.get(g, "#888"))
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(groups_present)+1))
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups_present], fontsize=8.5)
    ax.set_ylabel("Muscle Volume (cm³)")

    row = kw_df.loc[muscle] if muscle in kw_df.index else None
    if row is not None:
        h_stat = row["H_stat"]
        p_val  = row["p_value"]
        ax.set_title(f"{muscle} — Kruskal-Wallis\nH={h_stat:.3f}, p={p_val:.4f} {sig_stars(p_val)}",
                     fontweight="bold")

    # Pairwise annotations (Dunn's approach with Bonferroni)
    y_max = max(max(d) for d in data_g if len(d) > 0)
    step  = (y_max - df_m[TARGET].min()) * 0.08
    y_ann = y_max + step
    pairs = [(i, j) for i in range(len(groups_present)) for j in range(i+1, len(groups_present))]
    n_pairs = len(pairs)
    for pi, (i, j) in enumerate(pairs):
        if len(data_g[i]) < 2 or len(data_g[j]) < 2:
            continue
        _, pv = stats.mannwhitneyu(data_g[i], data_g[j], alternative="two-sided")
        pv_bonf = min(pv * n_pairs, 1.0)
        if pv_bonf < 0.10:
            y_br = y_ann + pi * step
            ax.plot([i+1, i+1, j+1, j+1], [y_br-step*0.3, y_br, y_br, y_br-step*0.3],
                    "k-", linewidth=0.8)
            ax.text((i+j)/2 + 1, y_br, sig_stars(pv_bonf), ha="center", fontsize=9,
                    fontweight="bold")
    ax.set_ylim(top=y_ann + len(pairs)*step + step*2)

fig.suptitle("Kruskal-Wallis: Significant Group Differences in Muscle Volume\n(pairwise Bonferroni-corrected Mann-Whitney)",
             fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "fig12_kruskal_wallis")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: LME MODEL COMPARISON  (AIC/BIC bar chart + R² summary)
# ══════════════════════════════════════════════════════════════════════════════
print("[+1] LME Model Comparison")
models = ["parsimonious", "full"]
labels = ["Parsimonious\n(Spearman-sig PCs)", "Full\n(All retained PCs)"]
aics   = [lme_stats[m]["AIC"] for m in models]
bics   = [lme_stats[m]["BIC"] for m in models]
r2m    = [lme_stats[m]["R2_marginal"] for m in models]
r2c    = [lme_stats[m]["R2_conditional"] for m in models]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# AIC/BIC
ax = axes[0]
x  = np.array([0, 1])
w  = 0.3
ax.bar(x - w/2, aics, w, label="AIC", color="#1565C0", alpha=0.8)
ax.bar(x + w/2, bics, w, label="BIC", color="#E65100", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Information Criterion (lower is better)")
ax.set_title("Model Comparison: AIC / BIC", fontweight="bold")
ax.legend()
for i, (a, b) in enumerate(zip(aics, bics)):
    ax.text(i-w/2, a+2, f"{a:.1f}", ha="center", fontsize=8)
    ax.text(i+w/2, b+2, f"{b:.1f}", ha="center", fontsize=8)

# R²
ax = axes[1]
ax.bar(x - w/2, r2m, w, label="R² Marginal (fixed)", color="#4CAF50", alpha=0.8)
ax.bar(x + w/2, r2c, w, label="R² Conditional (fixed+random)", color="#9C27B0", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Pseudo R²")
ax.set_ylim(0, min(max(r2c)*1.25, 1.0))
ax.set_title("Model Comparison: Pseudo R²", fontweight="bold")
ax.legend(fontsize=8)
for i, (rm, rc) in enumerate(zip(r2m, r2c)):
    ax.text(i-w/2, rm+0.01, f"{rm:.3f}", ha="center", fontsize=8)
    ax.text(i+w/2, rc+0.01, f"{rc:.3f}", ha="center", fontsize=8)

fig.suptitle("LME Model Fit Summary", fontweight="bold")
plt.tight_layout()
save(fig, "fig13_model_comparison")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL FIGURES SAVED TO:")
print(f"  {PLOT_DIR}")
print("="*60)
figs = sorted([f for f in os.listdir(PLOT_DIR) if f.endswith(".png")])
for f in figs:
    sz = os.path.getsize(os.path.join(PLOT_DIR, f))
    print(f"  {f:50s}  {sz//1024:>4d} KB")
