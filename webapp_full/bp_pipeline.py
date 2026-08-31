"""
bp_pipeline.py
==============
Derive the 19 BP features expected by the PCA pipeline from user-supplied
blood pressure readings.

User inputs required
--------------------
  sbp_rest   : resting SBP (mmHg) before exercise
  dbp_rest   : resting DBP (mmHg) before exercise
  sbp_post   : SBP (mmHg) measured after the last exercise set
  dbp_post   : DBP (mmHg) measured after the last exercise set
  sbp_ex_series : list/array of SBP readings during exercise (≥2 values)
  dbp_ex_series : list/array of DBP readings during exercise (≥2 values)

All 19 derived features are returned as a flat dict.
"""

import numpy as np


def _arv(x: np.ndarray) -> float:
    """Average Real Variability — mean of absolute successive differences."""
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))))


def _sv(x: np.ndarray) -> float:
    """Standard deviation of the series."""
    return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan


def _vim(x: np.ndarray, b: float = 1.0) -> float:
    """
    Variability Independent of Mean.
    VIM = std(x) / mean(x)^b
    b=1 gives the Coefficient of Variation (standard approximation when training b is unknown).
    """
    m = np.mean(x)
    s = _sv(x)
    if m > 0 and not np.isnan(s):
        return float(s / (m ** b))
    return np.nan


def compute_bp_features(
    sbp_rest: float,
    dbp_rest: float,
    sbp_post: float,
    dbp_post: float,
    sbp_ex_series,
    dbp_ex_series,
) -> dict:
    """
    Derive all 19 BP features matching pca_bp_features.json.

    Parameters
    ----------
    sbp_rest, dbp_rest   : resting BP before exercise (mmHg)
    sbp_post, dbp_post   : BP after last exercise set (mmHg)
    sbp_ex_series        : list of SBP readings during exercise (≥2)
    dbp_ex_series        : list of DBP readings during exercise (≥2)

    Returns
    -------
    dict of 19 feature names → float values
    """
    sbp_e = np.array(sbp_ex_series, dtype=float)
    dbp_e = np.array(dbp_ex_series, dtype=float)

    # ── Resting (pre-exercise) ────────────────────────────────────────────────
    map_rest = (sbp_rest + 2 * dbp_rest) / 3.0
    pp_rest  = sbp_rest - dbp_rest

    # ── Post-exercise ─────────────────────────────────────────────────────────
    map_post = (sbp_post + 2 * dbp_post) / 3.0

    # ── Delta rest → post ─────────────────────────────────────────────────────
    delta_sbp = sbp_post - sbp_rest

    # ── Exercise series stats ─────────────────────────────────────────────────
    sbp_mean_e   = float(np.mean(sbp_e))
    dbp_mean_e   = float(np.mean(dbp_e))
    map_e        = (sbp_mean_e + 2 * dbp_mean_e) / 3.0

    delta_sbp_e  = float(np.max(sbp_e) - np.min(sbp_e)) if len(sbp_e) >= 2 else np.nan
    delta_dbp_e  = float(np.max(dbp_e) - np.min(dbp_e)) if len(dbp_e) >= 2 else np.nan

    arv_sbp = _arv(sbp_e)
    arv_dbp = _arv(dbp_e)
    sv_sbp  = _sv(sbp_e)
    sv_dbp  = _sv(dbp_e)
    vim_sbp = _vim(sbp_e)
    vim_dbp = _vim(dbp_e)

    return {
        "SBP_rest_pre_mmHg":  float(sbp_rest),
        "DBP_rest_pre_mmHg":  float(dbp_rest),
        "MAP_rest_pre_mmHg":  map_rest,
        "PP_rest_pre_mmHg":   pp_rest,
        "SBP_post_set_mmHg":  float(sbp_post),
        "DBP_post_set_mmHg":  float(dbp_post),
        "MAP_post_set_mmHg":  map_post,
        "delta_SBP_mmHg":     delta_sbp,
        "SBP_mean_e":         sbp_mean_e,
        "DBP_mean_e":         dbp_mean_e,
        "MAP_e":              map_e,
        "Delta_SBP_e":        delta_sbp_e,
        "Delta_DBP_e":        delta_dbp_e,
        "ARV_SBP_mmHg":       arv_sbp,
        "SV_SBP_mmHg":        sv_sbp,
        "ARV_DBP_mmHg":       arv_dbp,
        "SV_DBP_mmHg":        sv_dbp,
        "VIM_SBP_mmHg":       vim_sbp,
        "VIM_DBP_mmHg":       vim_dbp,
    }
