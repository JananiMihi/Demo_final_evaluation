"""
pca_transform.py
================
Load saved scaler parameters + PCA loadings, then project new HR/BP feature
dicts into PC space for use in the LME prediction formula.

Assets read (from ML Modelling directory):
  pca_hr_scaler_mean.npy    training HR feature means
  pca_hr_scaler_scale.npy   training HR feature std-devs
  pca_bp_scaler_mean.npy    training BP feature means
  pca_bp_scaler_scale.npy   training BP feature std-devs
  pca_hr_features.json      ordered list of 44 HR feature names (post log-transform)
  pca_bp_features.json      ordered list of 19 BP feature names
  pca_hr_loadings.csv       eigenvectors: shape (44, 17)
  pca_bp_loadings.csv       eigenvectors: shape (19, 8)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

_ML_DIR = Path(__file__).parent.parent / "ML Modelling"


def _load():
    hr_mean  = np.load(_ML_DIR / "pca_hr_scaler_mean.npy")
    hr_scale = np.load(_ML_DIR / "pca_hr_scaler_scale.npy")
    bp_mean  = np.load(_ML_DIR / "pca_bp_scaler_mean.npy")
    bp_scale = np.load(_ML_DIR / "pca_bp_scaler_scale.npy")

    with open(_ML_DIR / "pca_hr_features.json") as f:
        hr_features = json.load(f)
    with open(_ML_DIR / "pca_bp_features.json") as f:
        bp_features = json.load(f)

    hr_loadings = pd.read_csv(_ML_DIR / "pca_hr_loadings.csv", index_col=0)
    bp_loadings = pd.read_csv(_ML_DIR / "pca_bp_loadings.csv", index_col=0)

    return {
        "hr_mean": hr_mean, "hr_scale": hr_scale,
        "bp_mean": bp_mean, "bp_scale": bp_scale,
        "hr_features": hr_features, "bp_features": bp_features,
        "hr_loadings": hr_loadings.values,   # (44, 17)
        "bp_loadings": bp_loadings.values,   # (19,  8)
        "hr_pc_names": list(hr_loadings.columns),
        "bp_pc_names": list(bp_loadings.columns),
    }


_ASSETS = None

def _get_assets():
    global _ASSETS
    if _ASSETS is None:
        _ASSETS = _load()
    return _ASSETS


def _resolve_value(feat_name: str, raw_feats: dict) -> float:
    """
    Map a final feature name (possibly ending in _log) back to its raw name,
    apply log1p if needed, and return the value. Returns NaN if missing.
    """
    if feat_name.endswith("_log"):
        raw_name = feat_name[:-4]           # strip "_log"
        val = raw_feats.get(raw_name, np.nan)
        if val is None or np.isnan(val):
            return np.nan
        # Guard: log1p requires val > -1; raw HRV/BP values should be non-negative
        val = max(val, 0.0)
        return float(np.log1p(val))
    else:
        val = raw_feats.get(feat_name, np.nan)
        return float(val) if val is not None else np.nan


def project_hr(raw_hr_feats: dict) -> dict:
    """
    Project a raw HR feature dict → HR PC scores.

    Parameters
    ----------
    raw_hr_feats : dict from hrv_pipeline.compute_hrv_features()
                   Keys are base names like "EX_RMSSD_ms", "REST_SDNN_ms", etc.

    Returns
    -------
    dict  PC name → float score (e.g. {"HR_PC1": 0.43, "HR_PC2": -0.12, …})
    """
    a = _get_assets()
    vec = np.array([_resolve_value(f, raw_hr_feats) for f in a["hr_features"]], dtype=float)

    # Replace NaN with 0 in scaled space (= training mean in raw space)
    nan_mask = np.isnan(vec)
    vec_scaled = np.where(nan_mask, 0.0, (vec - a["hr_mean"]) / a["hr_scale"])

    scores = vec_scaled @ a["hr_loadings"]   # shape (17,)
    return dict(zip(a["hr_pc_names"], scores.tolist()))


def project_bp(raw_bp_feats: dict) -> dict:
    """
    Project a raw BP feature dict → BP PC scores.

    Parameters
    ----------
    raw_bp_feats : dict from bp_pipeline.compute_bp_features()

    Returns
    -------
    dict  PC name → float score (e.g. {"BP_PC1": 0.21, "BP_PC2": -0.55, …})
    """
    a = _get_assets()
    vec = np.array([_resolve_value(f, raw_bp_feats) for f in a["bp_features"]], dtype=float)

    nan_mask = np.isnan(vec)
    vec_scaled = np.where(nan_mask, 0.0, (vec - a["bp_mean"]) / a["bp_scale"])

    scores = vec_scaled @ a["bp_loadings"]   # shape (8,)
    return dict(zip(a["bp_pc_names"], scores.tolist()))


def project_all(raw_hr_feats: dict, raw_bp_feats: dict) -> dict:
    """Combine HR and BP projection into one PC score dict."""
    hr_pcs = project_hr(raw_hr_feats)
    bp_pcs = project_bp(raw_bp_feats)
    return {**hr_pcs, **bp_pcs}
