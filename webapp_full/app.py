"""
HRVBP-HyperNet Web Application — Full Pipeline
===============================================
Flask backend with both prediction modes:
  1. Group-proxy prediction  (POST /api/predict)        — demographics + activity group
  2. Full CV prediction      (POST /api/predict-full)   — RR file + BP readings + demographics

Run:
    cd webapp_full
    python app.py
Then open: http://localhost:5001
"""

from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
ML_DIR = BASE / "ML Modelling"

# ── Load datasets ──────────────────────────────────────────────────────────────
test_preds  = pd.read_csv(ML_DIR / "ml_results" / "test_predictions.csv")
model_perf  = pd.read_csv(ML_DIR / "ml_results" / "model_comparison_report.csv")
rmse_muscle = pd.read_csv(ML_DIR / "ml_results" / "rmse_per_muscle.csv",  index_col=0)
rmse_group  = pd.read_csv(ML_DIR / "ml_results" / "rmse_per_group.csv",   index_col=0)
full_data   = pd.read_csv(ML_DIR / "ml_dataset_full.csv")
lme_effects = pd.read_csv(ML_DIR / "lme_fixed_effects_parsimonious.csv")

VOL_MEAN = full_data["volume_corrected_cm3"].mean()
VOL_STD  = full_data["volume_corrected_cm3"].std()

COEFF = {row["Unnamed: 0"]: row["Estimate"] for _, row in lme_effects.iterrows()}

MUSCLE_COEFF = {
    "BB":  0.0,
    "BF":  COEFF.get("C(Muscle)[T.BF]",  0),
    "DL":  COEFF.get("C(Muscle)[T.DL]",  0),
    "FDS": COEFF.get("C(Muscle)[T.FDS]", 0),
    "GA":  COEFF.get("C(Muscle)[T.GA]",  0),
    "TA":  COEFF.get("C(Muscle)[T.TA]",  0),
    "TB":  COEFF.get("C(Muscle)[T.TB]",  0),
    "VL":  COEFF.get("C(Muscle)[T.VL]",  0),
}

PCA_COLS = ["HR_PC2","HR_PC3","HR_PC5","HR_PC9","HR_PC10",
            "BP_PC3","BP_PC4","BP_PC5","BP_PC6"]
GROUP_PCA_MEANS = full_data.groupby("Group")[PCA_COLS].mean()

MUSCLES = ["BB", "BF", "DL", "FDS", "GA", "TA", "TB", "VL"]
MUSCLE_LABELS = {
    "BB": "Biceps Brachii",    "BF": "Bicep Femoris",
    "DL": "Deltoid",           "FDS": "Flexor Digitorum Superficialis",
    "GA": "Gastrocnemius",     "TA": "Tibialis Anterior",
    "TB": "Triceps Brachii",   "VL": "Vastus Lateralis",
}
SHAPE_CLASS = {
    "BB": "Fusiform",  "TB": "Fusiform",
    "DL": "Pennate",   "GA": "Pennate",
    "FDS": "Unipennate","TA": "Unipennate",
    "VL": "Strap",     "BF": "Strap",
}
GROUP_LABELS = {
    "G1": "Pure Resistance Training",
    "G2": "Combined (RT + Aerobic)",
    "G3": "Aerobic Only",
    "G4": "Sedentary",
}


# ── LME prediction helper ─────────────────────────────────────────────────────

def _safe(val, digits=2, default=0.0):
    """Return rounded float, replacing NaN/None with default."""
    try:
        v = float(val) if val is not None else default
        return round(default if (v != v) else v, digits)  # v!=v is True only for NaN
    except (TypeError, ValueError):
        return default


def _lme_predict(muscle, age, bmi, gender, pca_scores: dict) -> float:
    """
    Predict muscle volume (cm³) using LME fixed effects.
    pca_scores must contain HR_PC2, HR_PC3, HR_PC5, HR_PC9, HR_PC10,
    BP_PC3, BP_PC4, BP_PC5, BP_PC6.
    """
    z = (
        COEFF.get("Intercept", 0)
        + MUSCLE_COEFF[muscle]
        + COEFF.get("HR_PC2",  0) * pca_scores.get("HR_PC2",  0)
        + COEFF.get("HR_PC3",  0) * pca_scores.get("HR_PC3",  0)
        + COEFF.get("HR_PC5",  0) * pca_scores.get("HR_PC5",  0)
        + COEFF.get("HR_PC9",  0) * pca_scores.get("HR_PC9",  0)
        + COEFF.get("HR_PC10", 0) * pca_scores.get("HR_PC10", 0)
        + COEFF.get("BP_PC3",  0) * pca_scores.get("BP_PC3",  0)
        + COEFF.get("BP_PC4",  0) * pca_scores.get("BP_PC4",  0)
        + COEFF.get("BP_PC5",  0) * pca_scores.get("BP_PC5",  0)
        + COEFF.get("BP_PC6",  0) * pca_scores.get("BP_PC6",  0)
        + COEFF.get("Gender_bin", 0) * gender
        + COEFF.get("Age", 0) * age
        + COEFF.get("BMI", 0) * bmi
    )
    vol = float(z * VOL_STD + VOL_MEAN)
    return max(8.0, min(vol, 300.0))


# ── Routes — static pages ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Routes — API ──────────────────────────────────────────────────────────────

@app.route("/api/dashboard")
def dashboard():
    return jsonify({
        "study": {"participants": 78, "groups": 4, "muscles": 8,
                  "test_size": 16, "train_size": 62},
        "segmentation": {"dice": 0.7983, "iou": 0.7083, "hd95": 22.57,
                         "precision": 0.8181, "recall": 0.8339},
        "best_model": {"name": "LME", "rmse": 32.589, "r2": 0.298,
                       "ccc": 0.509, "f1": 0.526, "rank_rho": 0.505},
    })


@app.route("/api/model-performance")
def model_performance():
    rows = []
    for _, r in model_perf.iterrows():
        rows.append({
            "model":     r["Model"],
            "rmse":      round(float(r["M1_RMSE"]),      3),
            "r2":        round(float(r["R2"]),            3),
            "ccc":       round(float(r["M2_CCC"]),        3),
            "f1":        round(float(r["M3_F1"]),         3),
            "rank_rho":  round(float(r["M4_RankRho"]),   3),
            "r2_within": round(float(r["M5_R2within"]),  3),
        })
    return jsonify(rows)


@app.route("/api/rmse-per-muscle")
def rmse_per_muscle():
    return jsonify({
        m: {c: round(float(rmse_muscle.loc[m, c]), 2) for c in rmse_muscle.columns}
        for m in rmse_muscle.index
    })


@app.route("/api/rmse-per-group")
def rmse_per_group():
    return jsonify({
        m: {c: round(float(rmse_group.loc[m, c]), 2) for c in rmse_group.columns}
        for m in rmse_group.index
    })


@app.route("/api/test-predictions")
def test_predictions():
    rows = []
    for _, r in test_preds.iterrows():
        rows.append({
            "participant": r["participant"],
            "group":       r["Group"],
            "muscle":      r["Muscle"],
            "shape_class": r["shape_class"],
            "actual":      round(float(r["volume_corrected_cm3"]), 2),
            "pred_lme":    round(float(r["pred_LME"]),             2),
            "pred_merf":   round(float(r["pred_MERF"]),            2),
            "pred_gpboost":round(float(r["pred_GPBoost"]),         2),
            "label":       MUSCLE_LABELS.get(r["Muscle"], r["Muscle"]),
        })
    return jsonify(rows)


@app.route("/api/muscle-averages")
def muscle_averages():
    result = []
    for muscle in MUSCLES:
        sub = full_data[full_data["Muscle"] == muscle]["volume_corrected_cm3"]
        result.append({
            "muscle": muscle,
            "label":  MUSCLE_LABELS[muscle],
            "shape":  SHAPE_CLASS[muscle],
            "mean":   round(sub.mean(), 1),
            "std":    round(sub.std(),  1),
        })
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict_group():
    """Group-proxy prediction — uses group-level PCA means."""
    data   = request.json
    age    = float(data.get("age",    30))
    bmi    = float(data.get("bmi",    23))
    gender = int(data.get("gender",    1))
    group  = data.get("group", "G1")

    if group not in GROUP_PCA_MEANS.index:
        return jsonify({"error": f"Unknown group '{group}'"}), 400

    pca_scores = GROUP_PCA_MEANS.loc[group].to_dict()
    predictions = {}
    for muscle in MUSCLES:
        vol = _lme_predict(muscle, age, bmi, gender, pca_scores)
        predictions[muscle] = {
            "volume_cm3": round(vol, 1),
            "label": MUSCLE_LABELS[muscle],
            "shape_class": SHAPE_CLASS[muscle],
        }

    group_avgs = {}
    for m in MUSCLES:
        v = full_data[(full_data["Group"] == group) &
                      (full_data["Muscle"] == m)]["volume_corrected_cm3"].mean()
        group_avgs[m] = None if (v is None or np.isnan(v)) else round(float(v), 1)

    return jsonify({
        "mode": "group_proxy",
        "predictions": predictions,
        "group": group,
        "group_label": GROUP_LABELS[group],
        "group_avgs": group_avgs,
        "note": (
            "CV signal estimated from your activity group's average cardiovascular profile. "
            "Upload an RR interval file for individualised prediction."
        ),
    })


@app.route("/api/predict-full", methods=["POST"])
def predict_full():
    """
    Full individual prediction from RR interval file + BP readings.
    Multipart form data:
      rr_file      : uploaded text file with RR intervals
      rest_beats   : int — how many beats are the REST segment
      age, bmi, gender, group : same as /api/predict
      sbp_rest, dbp_rest       : resting BP before exercise
      sbp_post, dbp_post       : BP after last exercise set
      sbp_ex_series            : comma-separated SBP during exercise
      dbp_ex_series            : comma-separated DBP during exercise
    """
    from hrv_pipeline import parse_rr_file, compute_hrv_features
    from bp_pipeline  import compute_bp_features
    from pca_transform import project_all

    # ── Parse form ────────────────────────────────────────────────────────────
    age        = float(request.form.get("age",    30))
    bmi        = float(request.form.get("bmi",    23))
    gender     = int(request.form.get("gender",    1))
    group      = request.form.get("group", "G1")
    rest_beats = int(request.form.get("rest_beats", 150))

    sbp_rest   = float(request.form.get("sbp_rest", 120))
    dbp_rest   = float(request.form.get("dbp_rest",  78))
    sbp_post   = float(request.form.get("sbp_post", 145))
    dbp_post   = float(request.form.get("dbp_post",  88))

    def parse_series(key, fallback):
        raw = request.form.get(key, "")
        try:
            vals = [float(v.strip()) for v in raw.replace(";", ",").split(",") if v.strip()]
            return vals if len(vals) >= 2 else fallback
        except Exception:
            return fallback

    sbp_ex = parse_series("sbp_ex_series", [sbp_rest + 20, sbp_rest + 25, sbp_rest + 22])
    dbp_ex = parse_series("dbp_ex_series", [dbp_rest + 5,  dbp_rest + 7,  dbp_rest + 6])

    # ── RR file ───────────────────────────────────────────────────────────────
    if "rr_file" not in request.files or request.files["rr_file"].filename == "":
        return jsonify({"error": "No RR interval file uploaded."}), 400

    try:
        rr_bytes = request.files["rr_file"].read()
        rri_ms   = parse_rr_file(rr_bytes)
    except Exception as e:
        return jsonify({"error": f"RR file parsing failed: {e}"}), 400

    # ── HRV features ──────────────────────────────────────────────────────────
    try:
        raw_hr = compute_hrv_features(rri_ms, rest_beats)
    except Exception as e:
        return jsonify({"error": f"HRV computation failed: {e}"}), 500

    # ── BP features ───────────────────────────────────────────────────────────
    try:
        raw_bp = compute_bp_features(sbp_rest, dbp_rest, sbp_post, dbp_post,
                                     sbp_ex, dbp_ex)
    except Exception as e:
        return jsonify({"error": f"BP feature computation failed: {e}"}), 500

    # ── PCA projection ────────────────────────────────────────────────────────
    try:
        pca_scores = project_all(raw_hr, raw_bp)
    except Exception as e:
        return jsonify({"error": f"PCA projection failed: {e}"}), 500

    # ── LME prediction ────────────────────────────────────────────────────────
    predictions = {}
    for muscle in MUSCLES:
        vol = _lme_predict(muscle, age, bmi, gender, pca_scores)
        predictions[muscle] = {
            "volume_cm3":  round(vol, 1),
            "label":       MUSCLE_LABELS[muscle],
            "shape_class": SHAPE_CLASS[muscle],
        }

    group_avgs = {}
    for m in MUSCLES:
        v = full_data[(full_data["Group"] == group) &
                      (full_data["Muscle"] == m)]["volume_corrected_cm3"].mean()
        group_avgs[m] = None if (v is None or np.isnan(v)) else round(float(v), 1)

    # Key PC scores used in prediction (for transparency display)
    key_pcs = {pc: round(pca_scores.get(pc, 0), 4) for pc in PCA_COLS}

    # Raw HRV summary to show user
    hrv_summary = {
        "rest_beats":       rest_beats,
        "total_beats":      len(rri_ms),
        "REST_RMSSD_ms":    _safe(raw_hr.get("REST_RMSSD_ms"),    2),
        "REST_Mean_HR_bpm": _safe(raw_hr.get("REST_Mean_HR_bpm"), 1),
        "EX_RMSSD_ms":      _safe(raw_hr.get("EX_RMSSD_ms"),      2),
        "EX_Mean_HR_bpm":   _safe(raw_hr.get("EX_Mean_HR_bpm"),   1),
        "EX_LF_HF_ratio":   _safe(raw_hr.get("EX_LF_HF_ratio_FFT"), 3),
        "REST_DFA_alpha1":  _safe(raw_hr.get("REST_DFA_alpha1"),   3),
    }

    return jsonify({
        "mode":         "full_cv",
        "predictions":  predictions,
        "group":        group,
        "group_label":  GROUP_LABELS.get(group, group),
        "group_avgs":   group_avgs,
        "key_pcs":      key_pcs,
        "hrv_summary":  hrv_summary,
        "note": (
            "Prediction uses your individual cardiovascular signal profile "
            f"({len(rri_ms)} RR intervals) projected through the trained PCA → LME pipeline."
        ),
    })


@app.route("/api/predict-live", methods=["POST"])
def predict_live():
    """
    Live-band prediction — RR intervals supplied as JSON array (from Web Bluetooth).
    Body (JSON):
      rr_intervals   : list of RR intervals in ms
      rest_beats     : int — split point between REST and EXERCISE
      age, bmi, gender, group : demographics
      sbp_rest, dbp_rest, sbp_post, dbp_post : BP readings (float)
      sbp_ex_series, dbp_ex_series : lists of exercise-series BP values
    """
    from hrv_pipeline import compute_hrv_features
    from bp_pipeline  import compute_bp_features
    from pca_transform import project_all

    body = request.json or {}
    rr_list = body.get("rr_intervals", [])
    if len(rr_list) < 60:
        return jsonify({"error": f"Only {len(rr_list)} RR intervals received — need at least 60."}), 400

    rri_ms     = np.array(rr_list, dtype=float)
    rest_beats = int(body.get("rest_beats", 150))
    age        = float(body.get("age",    30))
    bmi        = float(body.get("bmi",    23))
    gender     = int(body.get("gender",    1))
    group      = body.get("group", "G1")

    sbp_rest = float(body.get("sbp_rest", 120))
    dbp_rest = float(body.get("dbp_rest",  78))
    sbp_post = float(body.get("sbp_post", 145))
    dbp_post = float(body.get("dbp_post",  88))
    sbp_ex   = body.get("sbp_ex_series", [sbp_rest + 20, sbp_rest + 25, sbp_rest + 22])
    dbp_ex   = body.get("dbp_ex_series", [dbp_rest + 5,  dbp_rest + 7,  dbp_rest + 6])

    try:
        raw_hr = compute_hrv_features(rri_ms, rest_beats)
    except Exception as e:
        return jsonify({"error": f"HRV computation failed: {e}"}), 500

    try:
        raw_bp = compute_bp_features(sbp_rest, dbp_rest, sbp_post, dbp_post, sbp_ex, dbp_ex)
    except Exception as e:
        return jsonify({"error": f"BP feature computation failed: {e}"}), 500

    try:
        pca_scores = project_all(raw_hr, raw_bp)
    except Exception as e:
        return jsonify({"error": f"PCA projection failed: {e}"}), 500

    predictions = {}
    for muscle in MUSCLES:
        vol = _lme_predict(muscle, age, bmi, gender, pca_scores)
        predictions[muscle] = {
            "volume_cm3":  round(vol, 1),
            "label":       MUSCLE_LABELS[muscle],
            "shape_class": SHAPE_CLASS[muscle],
        }

    group_avgs = {}
    for m in MUSCLES:
        v = full_data[(full_data["Group"] == group) &
                      (full_data["Muscle"] == m)]["volume_corrected_cm3"].mean()
        group_avgs[m] = None if (v is None or np.isnan(v)) else round(float(v), 1)

    key_pcs = {pc: round(pca_scores.get(pc, 0), 4) for pc in PCA_COLS}
    hrv_summary = {
        "rest_beats":       rest_beats,
        "total_beats":      len(rri_ms),
        "REST_RMSSD_ms":    _safe(raw_hr.get("REST_RMSSD_ms"),       2),
        "REST_Mean_HR_bpm": _safe(raw_hr.get("REST_Mean_HR_bpm"),    1),
        "EX_RMSSD_ms":      _safe(raw_hr.get("EX_RMSSD_ms"),         2),
        "EX_Mean_HR_bpm":   _safe(raw_hr.get("EX_Mean_HR_bpm"),      1),
        "EX_LF_HF_ratio":   _safe(raw_hr.get("EX_LF_HF_ratio_FFT"), 3),
        "REST_DFA_alpha1":  _safe(raw_hr.get("REST_DFA_alpha1"),     3),
    }

    return jsonify({
        "mode":        "live_band",
        "predictions": predictions,
        "group":       group,
        "group_label": GROUP_LABELS.get(group, group),
        "group_avgs":  group_avgs,
        "key_pcs":     key_pcs,
        "hrv_summary": hrv_summary,
        "note": (
            f"Live recording: {len(rri_ms)} RR intervals via Web Bluetooth "
            f"(REST {rest_beats} beats) → PCA → LME pipeline."
        ),
    })


if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  HRVBP-HyperNet Dashboard — Full CV Pipeline")
    print("  University of Ruhuna — EG/2021")
    print("=" * 58)
    print(f"  Dataset rows  : {len(full_data)}")
    print(f"  VOL_MEAN      : {VOL_MEAN:.1f} cm³")
    print(f"  VOL_STD       : {VOL_STD:.1f} cm³")
    print("=" * 58)
    print("  Open: http://localhost:5001")
    print("=" * 58 + "\n")
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=5001)
