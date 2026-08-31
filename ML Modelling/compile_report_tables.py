"""
compile_report_tables.py
Reads all computed outputs and writes a single Excel workbook with one sheet
per paper table (4.4 – 4.21), ready to copy-paste into the report.
"""

import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR   = os.path.join(WORK_DIR, "ml_results")
OUT_PATH = os.path.join(WORK_DIR, "Report_Tables_All.xlsx")

TARGET = "volume_corrected_cm3"

# ── Load raw outputs ──────────────────────────────────────────────────────────
pca_json  = json.load(open(os.path.join(WORK_DIR, "pca_summary.json")))
vif_df    = pd.read_csv(os.path.join(WORK_DIR, "vif_summary.csv"))
sp_full   = pd.read_csv(os.path.join(WORK_DIR, "spearman_results.csv"))
sp_pool   = pd.read_csv(os.path.join(WORK_DIR, "spearman_pooled.csv"))
sp_pivot  = pd.read_csv(os.path.join(WORK_DIR, "spearman_per_muscle_pivot.csv"), index_col=0)
lme_stats = json.load(open(os.path.join(WORK_DIR, "lme_stats.json")))
lme_pars  = pd.read_csv(os.path.join(WORK_DIR, "lme_fixed_effects_parsimonious.csv"), index_col=0)
lme_full  = pd.read_csv(os.path.join(WORK_DIR, "lme_fixed_effects_full.csv"), index_col=0)
desc_df   = pd.read_csv(os.path.join(WORK_DIR, "descriptive_stats.csv"), header=[0,1], index_col=0)
vol_grp   = pd.read_csv(os.path.join(WORK_DIR, "volume_by_group_muscle.csv"))
kw_df     = pd.read_csv(os.path.join(WORK_DIR, "kruskal_wallis.csv"))
ml_report = pd.read_csv(os.path.join(ML_DIR, "model_comparison_report.csv"))
ml_muscle = pd.read_csv(os.path.join(ML_DIR, "rmse_per_muscle.csv"))
ml_group  = pd.read_csv(os.path.join(ML_DIR, "rmse_per_group.csv"))
ext_met   = json.load(open(os.path.join(ML_DIR, "extended_metrics.json")))
test_preds = pd.read_csv(os.path.join(ML_DIR, "test_predictions.csv"))

# ── Compute additional metrics (r, rho) for ML models ─────────────────────────
def pearson_r(df, yt_col, yp_col):
    mask = df[yt_col].notna() & df[yp_col].notna()
    return float(stats.pearsonr(df.loc[mask, yt_col], df.loc[mask, yp_col])[0])

def spearman_rho(df, yt_col, yp_col):
    mask = df[yt_col].notna() & df[yp_col].notna()
    return float(stats.spearmanr(df.loc[mask, yt_col], df.loc[mask, yp_col])[0])

extra_metrics = {}
for model in ["LME", "MERF", "GPBoost", "Ensemble"]:
    pred_col = f"pred_{model}"
    if pred_col in test_preds.columns:
        r   = pearson_r(test_preds, TARGET, pred_col)
        rho = spearman_rho(test_preds, TARGET, pred_col)
        extra_metrics[model] = {"Pearson_r": round(r, 4), "Spearman_rho": round(rho, 4)}

# ── Build per-muscle R² for MERF and GPBoost (for Tables 4.10 / 4.13) ─────────
def per_muscle_r2(df, yt_col, yp_col):
    out = {}
    for m in df["Muscle"].unique():
        sub = df[df["Muscle"] == m]
        yt  = sub[yt_col].values
        yp  = sub[yp_col].values if yp_col in sub.columns else np.full(len(yt), np.nan)
        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - yt.mean())**2)
        out[m] = round(1 - ss_res/ss_tot, 4) if ss_tot > 0 else np.nan
    return out

muscle_r2 = {}
for model in ["LME", "MERF", "GPBoost", "Ensemble"]:
    pred_col = f"pred_{model}"
    if pred_col in test_preds.columns:
        muscle_r2[model] = per_muscle_r2(test_preds, TARGET, pred_col)

def per_group_r2(df, yt_col, yp_col):
    out = {}
    for g in df["Group"].unique():
        sub = df[df["Group"] == g]
        yt  = sub[yt_col].values
        yp  = sub[yp_col].values if yp_col in sub.columns else np.full(len(yt), np.nan)
        ss_res = np.sum((yt - yp)**2)
        ss_tot = np.sum((yt - yt.mean())**2)
        out[g] = round(1 - ss_res/ss_tot, 4) if ss_tot > 0 else np.nan
    return out

group_r2 = {}
for model in ["LME", "MERF", "GPBoost", "Ensemble"]:
    pred_col = f"pred_{model}"
    if pred_col in test_preds.columns:
        group_r2[model] = per_group_r2(test_preds, TARGET, pred_col)

# ── Assemble all tables ────────────────────────────────────────────────────────
print("Assembling report tables...")

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:

    # ── Table 4.4: Domain-wise PCA Summary ────────────────────────────────────
    t44 = pd.DataFrame([
        {"Domain": "HR (HRV & HR Dynamics)",
         "Raw Features": pca_json["HR_raw_features"] if "HR_raw_features" in pca_json else len(pca_json.get("HR_variance_explained",[])),
         "PCs Retained": pca_json["HR_components"],
         "Variance Explained (%)": round(sum(pca_json["HR_variance_explained"]), 1)},
        {"Domain": "BP (BPV & Blood Pressure)",
         "Raw Features": pca_json["BP_raw_features"] if "BP_raw_features" in pca_json else len(pca_json.get("BP_variance_explained",[])),
         "PCs Retained": pca_json["BP_components"],
         "Variance Explained (%)": round(sum(pca_json["BP_variance_explained"]), 1)},
    ])
    t44.to_excel(writer, sheet_name="T4.4_PCA_Summary", index=False)

    # ── Table 4.5: PCA Variance per component ────────────────────────────────
    t45_rows = []
    for i, v in enumerate(pca_json["HR_variance_explained"]):
        t45_rows.append({"Domain": "HR", "Component": f"HR_PC{i+1}", "Variance_Explained_%": v})
    for i, v in enumerate(pca_json["BP_variance_explained"]):
        t45_rows.append({"Domain": "BP", "Component": f"BP_PC{i+1}", "Variance_Explained_%": v})
    pd.DataFrame(t45_rows).to_excel(writer, sheet_name="T4.5_PCA_Variance", index=False)

    # ── Table 4.6: VIF Filtering Summary ──────────────────────────────────────
    n_checked  = len(vif_df)
    n_removed  = len(vif_df[vif_df["Status"] == "Removed"])
    n_retained = len(vif_df[vif_df["Status"] == "Retained"])
    t46 = pd.DataFrame([
        {"": "Features checked",  "Count": n_checked},
        {"": "Features dropped",  "Count": n_removed},
        {"": "Features retained", "Count": n_retained},
    ])
    t46.to_excel(writer, sheet_name="T4.6_VIF_Summary", index=False)

    # ── Table 4.7: Dropped Features (VIF > 10) ────────────────────────────────
    dropped = vif_df[vif_df["Status"] == "Removed"][["Feature", "VIF"]]
    dropped.to_excel(writer, sheet_name="T4.7_VIF_Dropped", index=False)

    # ── VIF Full detail ────────────────────────────────────────────────────────
    vif_df.to_excel(writer, sheet_name="T4.7b_VIF_Full", index=False)

    # ── Table 4.8: Spearman Screening ──────────────────────────────────────────
    sig_sp = sp_full[sp_full["Significant"]].sort_values("rho", key=abs, ascending=False).copy()
    sig_sp = sig_sp[["Component", "Muscle", "rho", "p_value", "n", "Significant"]]
    sig_sp["Selected"] = "✓"
    t48 = pd.DataFrame({
        "Note": [f"Retained: {len(sig_sp[sig_sp['Significant']].drop_duplicates('Component'))} components with ≥1 significant muscle",
                 f"Total pairs tested: {len(sp_full)}",
                 f"Significant pairs (|rho|≥0.30, p<0.05): {len(sig_sp)}"],
    })
    t48.to_excel(writer, sheet_name="T4.8_Spearman_Note", index=False)
    sig_sp.to_excel(writer, sheet_name="T4.8_Spearman_Screening", index=False)
    sp_pool.to_excel(writer, sheet_name="T4.8_Spearman_Pooled", index=False)

    # ── Table 4.9: LME Performance ────────────────────────────────────────────
    lme_row = ml_report[ml_report["Model"] == "LME"].iloc[0]
    r_lme   = extra_metrics.get("LME", {})
    t49 = pd.DataFrame([
        {"Metric": "RMSE (cm³)",       "Parsimonious_LME": round(lme_row["M1_RMSE"], 4),
         "Description": "Root Mean Squared Error on test set"},
        {"Metric": "MSE (cm³²)",       "Parsimonious_LME": round(lme_row["M1_RMSE"]**2, 4),
         "Description": "Mean Squared Error"},
        {"Metric": "R²",               "Parsimonious_LME": round(lme_row["R2"], 4),
         "Description": "Coefficient of determination"},
        {"Metric": "CCC (Lin's)",       "Parsimonious_LME": round(lme_row["M2_CCC"], 4),
         "Description": "Concordance Correlation Coefficient (M2)"},
        {"Metric": "Pearson r",         "Parsimonious_LME": r_lme.get("Pearson_r", "N/A"),
         "Description": "Pearson correlation"},
        {"Metric": "Spearman ρ",        "Parsimonious_LME": r_lme.get("Spearman_rho", "N/A"),
         "Description": "Spearman correlation"},
        {"Metric": "Bland-Altman Bias", "Parsimonious_LME": round(lme_row["M2_Bias"], 4),
         "Description": "Mean bias (M2)"},
        {"Metric": "LoA Low",           "Parsimonious_LME": round(lme_row["M2_LoA_lo"], 4),
         "Description": "95% Limits of Agreement lower"},
        {"Metric": "LoA High",          "Parsimonious_LME": round(lme_row["M2_LoA_hi"], 4),
         "Description": "95% Limits of Agreement upper"},
        {"Metric": "F1 (Responder)",    "Parsimonious_LME": round(lme_row["M3_F1"], 4),
         "Description": "F1 score for Responder/Non-Responder classification (M3)"},
        {"Metric": "Rank Rho (M4)",     "Parsimonious_LME": round(lme_row["M4_RankRho"], 4),
         "Description": "Cross-muscle ranking correlation"},
        {"Metric": "R²_within (M5)",    "Parsimonious_LME": round(lme_row["M5_R2within"], 4),
         "Description": "Within-group stratification R²"},
        {"Metric": "R²_marginal",       "Parsimonious_LME": lme_stats.get("parsimonious", {}).get("R2_marginal", "N/A"),
         "Description": "LME pseudo R² (fixed effects only)"},
        {"Metric": "R²_conditional",    "Parsimonious_LME": lme_stats.get("parsimonious", {}).get("R2_conditional", "N/A"),
         "Description": "LME pseudo R² (fixed + random effects)"},
        {"Metric": "ICC",               "Parsimonious_LME": lme_stats.get("parsimonious", {}).get("ICC", "N/A"),
         "Description": "Intraclass Correlation Coefficient (participant)"},
        {"Metric": "AIC",               "Parsimonious_LME": lme_stats.get("parsimonious", {}).get("AIC", "N/A"),
         "Description": "Akaike Information Criterion"},
        {"Metric": "BIC",               "Parsimonious_LME": lme_stats.get("parsimonious", {}).get("BIC", "N/A"),
         "Description": "Bayesian Information Criterion"},
    ])
    t49.to_excel(writer, sheet_name="T4.9_LME_Performance", index=False)

    # LME fixed effects detail
    lme_pars.to_excel(writer, sheet_name="T4.9b_LME_FixedEffects")

    # ── Tables 4.10 & 4.11: MERF Performance ─────────────────────────────────
    merf_row = ml_report[ml_report["Model"] == "MERF"].iloc[0]
    r_merf   = extra_metrics.get("MERF", {})
    t411 = pd.DataFrame([
        {"Metric": "RMSE (cm³)",      "Value": round(merf_row["M1_RMSE"], 4)},
        {"Metric": "MSE (cm³²)",      "Value": round(merf_row["M1_RMSE"]**2, 4)},
        {"Metric": "R²",              "Value": round(merf_row["R2"], 4)},
        {"Metric": "Pearson r",       "Value": r_merf.get("Pearson_r", "N/A")},
        {"Metric": "Spearman ρ",      "Value": r_merf.get("Spearman_rho", "N/A")},
        {"Metric": "CCC (Lin's)",     "Value": round(merf_row["M2_CCC"], 4)},
        {"Metric": "Bias (Bland-Alt)","Value": round(merf_row["M2_Bias"], 4)},
        {"Metric": "LoA Low",         "Value": round(merf_row["M2_LoA_lo"], 4)},
        {"Metric": "LoA High",        "Value": round(merf_row["M2_LoA_hi"], 4)},
        {"Metric": "F1 (Responder)",  "Value": round(merf_row["M3_F1"], 4)},
        {"Metric": "Rank Rho (M4)",   "Value": round(merf_row["M4_RankRho"], 4)},
        {"Metric": "R²_within (M5)",  "Value": round(merf_row["M5_R2within"], 4)},
        {"Metric": "Training Time (s)","Value": "See ml_modelling output"},
    ])
    t411.to_excel(writer, sheet_name="T4.11_MERF_Overall", index=False)

    # Per-muscle MERF (Table 4.10) — ml_muscle rows=models, cols=muscles
    muscles = [c for c in ml_muscle.columns if c != "Unnamed: 0"]
    merf_m_row = ml_muscle[ml_muscle["Unnamed: 0"] == "MERF"].iloc[0]
    t410_rows = []
    for m in muscles:
        t410_rows.append({"Muscle": m,
                          "MERF_RMSE": round(float(merf_m_row[m]), 4),
                          "MERF_R2": muscle_r2.get("MERF", {}).get(m, "N/A")})
    pd.DataFrame(t410_rows).to_excel(writer, sheet_name="T4.10_MERF_PerMuscle", index=False)

    # Per-group MERF (Table 4.12)
    groups = [c for c in ml_group.columns if c != "Unnamed: 0"]
    merf_g_row = ml_group[ml_group["Unnamed: 0"] == "MERF"].iloc[0]
    t412_rows = []
    for g in groups:
        t412_rows.append({"Group": g,
                          "MERF_RMSE": round(float(merf_g_row[g]), 4),
                          "MERF_R2": group_r2.get("MERF", {}).get(g, "N/A")})
    pd.DataFrame(t412_rows).to_excel(writer, sheet_name="T4.12_MERF_PerGroup", index=False)

    # ── Tables 4.13-4.15: GPBoost Performance ────────────────────────────────
    gpb_row  = ml_report[ml_report["Model"] == "GPBoost"].iloc[0]
    r_gpb    = extra_metrics.get("GPBoost", {})
    t414 = pd.DataFrame([
        {"Metric": "RMSE (cm³)",      "Value": round(gpb_row["M1_RMSE"], 4)},
        {"Metric": "MSE (cm³²)",      "Value": round(gpb_row["M1_RMSE"]**2, 4)},
        {"Metric": "R²",              "Value": round(gpb_row["R2"], 4)},
        {"Metric": "Pearson r",       "Value": r_gpb.get("Pearson_r", "N/A")},
        {"Metric": "Spearman ρ",      "Value": r_gpb.get("Spearman_rho", "N/A")},
        {"Metric": "CCC (Lin's)",     "Value": round(gpb_row["M2_CCC"], 4)},
        {"Metric": "Bias (Bland-Alt)","Value": round(gpb_row["M2_Bias"], 4)},
        {"Metric": "LoA Low",         "Value": round(gpb_row["M2_LoA_lo"], 4)},
        {"Metric": "LoA High",        "Value": round(gpb_row["M2_LoA_hi"], 4)},
        {"Metric": "F1 (Responder)",  "Value": round(gpb_row["M3_F1"], 4)},
        {"Metric": "Rank Rho (M4)",   "Value": round(gpb_row["M4_RankRho"], 4)},
        {"Metric": "R²_within (M5)",  "Value": round(gpb_row["M5_R2within"], 4)},
        {"Metric": "Training Time (s)","Value": "See ml_modelling output"},
    ])
    t414.to_excel(writer, sheet_name="T4.14_GPBoost_Overall", index=False)

    gpb_m_row = ml_muscle[ml_muscle["Unnamed: 0"] == "GPBoost"].iloc[0]
    t413_rows = []
    for m in muscles:
        t413_rows.append({"Muscle": m,
                          "GPBoost_RMSE": round(float(gpb_m_row[m]), 4),
                          "GPBoost_R2": muscle_r2.get("GPBoost", {}).get(m, "N/A")})
    pd.DataFrame(t413_rows).to_excel(writer, sheet_name="T4.13_GPBoost_PerMuscle", index=False)

    gpb_g_row = ml_group[ml_group["Unnamed: 0"] == "GPBoost"].iloc[0]
    t415_rows = []
    for g in groups:
        t415_rows.append({"Group": g,
                          "GPBoost_RMSE": round(float(gpb_g_row[g]), 4),
                          "GPBoost_R2": group_r2.get("GPBoost", {}).get(g, "N/A")})
    pd.DataFrame(t415_rows).to_excel(writer, sheet_name="T4.15_GPBoost_PerGroup", index=False)

    # ── Full model comparison (M1-M5 side by side) ────────────────────────────
    comp_rows = []
    for model in ["LME", "MERF", "GPBoost", "Ensemble"]:
        row = ml_report[ml_report["Model"] == model].iloc[0].to_dict()
        row.update(extra_metrics.get(model, {}))
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_excel(writer, sheet_name="T_All_Models_Comparison", index=False)

    # ── M9-M11 Extended metrics ────────────────────────────────────────────────
    ext_rows = [
        {"Metric": "M9 Ensemble Gain (%)",          "Value": ext_met.get("M9_ensemble_gain_pct","N/A")},
        {"Metric": "M9 Delta RMSE (cm³)",            "Value": ext_met.get("M9_delta_rmse_cm3","N/A")},
        {"Metric": "M10 Delta R² (Random Effects)",  "Value": ext_met.get("M10_delta_R2_RE","N/A")},
        {"Metric": "M11 Missingness slope γ",        "Value": ext_met.get("M11_missingness_gamma","N/A")},
    ]
    for rate, val in ext_met.get("M11_rmse_by_rate", {}).items():
        ext_rows.append({"Metric": f"M11 RMSE at {rate}% missingness", "Value": val})
    pd.DataFrame(ext_rows).to_excel(writer, sheet_name="T_M9_M10_M11_Extended", index=False)

    # ── Table 4.20: B2 Sample Efficiency ──────────────────────────────────────
    b2_path = os.path.join(ML_DIR, "b2_sample_efficiency.csv")
    if os.path.exists(b2_path):
        pd.read_csv(b2_path).to_excel(writer, sheet_name="T4.20_B2_SampleEff", index=False)
    else:
        b2_note = pd.DataFrame([{"Note": "B2 results computed — see b2_results.json in ml_results/",
                                  "Train sizes available": "N=20, 30, 40, 50 (dataset limited to 63 participants)"}])
        b2_note.to_excel(writer, sheet_name="T4.20_B2_SampleEff", index=False)

    # Load B2 from JSON if CSV not available
    b2_json_path = os.path.join(ML_DIR, "b2_results.json")
    if os.path.exists(b2_json_path):
        b2_raw = json.load(open(b2_json_path))
        b2_rows = []
        for N_str, model_dict in b2_raw.items():
            for model, metrics in model_dict.items():
                m1_vals = [v for v in metrics.get("M1",[]) if not (v is None or (isinstance(v,float) and np.isnan(v)))]
                m5_vals = [v for v in metrics.get("M5",[]) if not (v is None or (isinstance(v,float) and np.isnan(v)))]
                b2_rows.append({
                    "N_train": int(N_str),
                    "Method": model,
                    "M1_RMSE_mean": round(float(np.mean(m1_vals)),4) if m1_vals else None,
                    "M1_RMSE_std":  round(float(np.std(m1_vals)),4)  if m1_vals else None,
                    "M5_R2w_mean":  round(float(np.mean(m5_vals)),4) if m5_vals else None,
                    "M5_R2w_std":   round(float(np.std(m5_vals)),4)  if m5_vals else None,
                })
        pd.DataFrame(b2_rows).to_excel(writer, sheet_name="T4.20_B2_SampleEff", index=False)

    # ── Table 4.21: B3 Annotation Efficiency (not computable without image data)
    t421 = pd.DataFrame([
        {"Note": "Table 4.21 (B3) requires RST-UNet segmentation model training",
         "Status": "REQUIRES ULTRASOUND IMAGE DATA",
         "Details": "B3 tests RST-UNet Dice and volume error at K=10,25,50,100,full annotation budgets. "
                    "This requires physiotherapist-annotated ultrasound mask images which are not "
                    "part of the tabular dataset. Cannot be computed from Full_Data_Set_preprocessed.xlsx."},
    ])
    t421.to_excel(writer, sheet_name="T4.21_B3_NOTE", index=False)

    # ── Descriptive Statistics ────────────────────────────────────────────────
    desc_df.to_excel(writer, sheet_name="T_Descriptive_Stats")
    vol_grp.to_excel(writer, sheet_name="T_Volume_by_Group_Muscle", index=False)
    kw_df.to_excel(writer, sheet_name="T_Kruskal_Wallis", index=False)

print(f"\nReport tables workbook saved → {OUT_PATH}")
print("\n" + "=" * 70)
print("ALL TABLE VALUES SUMMARY")
print("=" * 70)

# Print concise summary of key values
print("\n📊 Table 4.4 — PCA Summary:")
print(f"   HR: {pca_json['HR_components']} PCs retained from raw features, {sum(pca_json['HR_variance_explained']):.1f}% variance")
print(f"   BP: {pca_json['BP_components']} PCs retained from raw features, {sum(pca_json['BP_variance_explained']):.1f}% variance")

print("\n📊 Table 4.6 — VIF Filtering:")
print(f"   Checked: {len(vif_df)} | Dropped: {len(vif_df[vif_df['Status']=='Removed'])} | Retained: {len(vif_df[vif_df['Status']=='Retained'])}")

print("\n📊 Table 4.7 — VIF-Dropped Features:")
for _, r in vif_df[vif_df["Status"]=="Removed"].iterrows():
    print(f"   {r['Feature']}  (VIF={r['VIF']:.2f})")

print("\n📊 Table 4.8 — Spearman Screening:")
sig_sp_out = sp_full[sp_full["Significant"]].sort_values("rho", key=abs, ascending=False)
print(f"   {len(sig_sp_out)} significant pairs out of {len(sp_full)} tested")
print(f"   Significant components: {sorted(sig_sp_out['Component'].unique())}")

print("\n📊 Table 4.9 — LME Model Performance (test set):")
lme = ml_report[ml_report["Model"]=="LME"].iloc[0]
print(f"   RMSE={lme['M1_RMSE']:.4f} cm³  |  R²={lme['R2']:.4f}  |  CCC={lme['M2_CCC']:.4f}")
print(f"   Pseudo R²_marginal={lme_stats['parsimonious']['R2_marginal']}  |  R²_conditional={lme_stats['parsimonious']['R2_conditional']}")

print("\n📊 Tables 4.10-4.12 — MERF Performance:")
merf = ml_report[ml_report["Model"]=="MERF"].iloc[0]
r_m = extra_metrics.get("MERF",{})
print(f"   RMSE={merf['M1_RMSE']:.4f} cm³  |  R²={merf['R2']:.4f}  |  r={r_m.get('Pearson_r','N/A')}  |  ρ={r_m.get('Spearman_rho','N/A')}")
print(f"   CCC={merf['M2_CCC']:.4f}  |  F1={merf['M3_F1']:.4f}  |  R²_within={merf['M5_R2within']:.4f}")

print("\n📊 Tables 4.13-4.15 — GPBoost Performance:")
gpb = ml_report[ml_report["Model"]=="GPBoost"].iloc[0]
r_g = extra_metrics.get("GPBoost",{})
print(f"   RMSE={gpb['M1_RMSE']:.4f} cm³  |  R²={gpb['R2']:.4f}  |  r={r_g.get('Pearson_r','N/A')}  |  ρ={r_g.get('Spearman_rho','N/A')}")
print(f"   CCC={gpb['M2_CCC']:.4f}  |  F1={gpb['M3_F1']:.4f}  |  R²_within={gpb['M5_R2within']:.4f}")

print("\n📊 M9-M11 Extended Metrics:")
print(f"   M9 Ensemble Gain: {ext_met['M9_ensemble_gain_pct']}%  |  ΔR²_RE (M10): {ext_met['M10_delta_R2_RE']}")
print(f"   M11 slope γ: {ext_met['M11_missingness_gamma']} cm³/1%")

print("\n📊 Table 4.20 — B2 Sample Efficiency (computed, see T4.20 sheet):")
print(f"   Dataset limited to 63 participants → train sizes N=20,30,40,50")
print(f"   Full table in Report_Tables_All.xlsx → T4.20_B2_SampleEff")

print("\n⚠️  Table 4.21 — B3 Annotation Efficiency:")
print("   CANNOT COMPUTE — requires ultrasound mask image data (RST-UNet training)")
print("   This table must be filled from segmentation experiments separately.")
