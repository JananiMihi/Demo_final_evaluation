"""
hrv_pipeline.py
===============
Compute the 44 HRV raw features expected by the PCA pipeline from a user-supplied
RR-interval file.

Accepted file formats:
  • One RR value per line (milliseconds OR seconds auto-detected)
  • Space / comma-separated values on a single line or multiple lines
  • Polar H10 CSV export (auto-detects RR column header)

REST / EXERCISE split:
  • rest_beats (int) — first N beats are the REST segment; remainder = EXERCISE

Feature list (44 total) matches pca_hr_features.json exactly.
"""

import io
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── neurokit2 ─────────────────────────────────────────────────────────────────
try:
    import neurokit2 as nk
    _NK_OK = True
except ImportError:
    _NK_OK = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_rr_file(content: bytes) -> np.ndarray:
    """
    Parse uploaded file bytes → 1-D numpy array of RR intervals in milliseconds.
    Auto-detects ms vs seconds (values < 3 → seconds, multiply by 1000).
    Handles plain text and Polar CSV exports.
    """
    text = content.decode("utf-8", errors="ignore")

    # Polar H10 CSV?  Look for an RR interval column header.
    if "RR interval" in text or "rr_interval" in text.lower():
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        rr_col = next((c for c in df.columns
                       if "rr" in c.lower() and "interval" in c.lower()), None)
        if rr_col:
            rri = df[rr_col].dropna().to_numpy(dtype=float)
            if rri.mean() < 3:
                rri = rri * 1000.0
            return rri

    # Plain text: collect all numeric tokens
    import re
    tokens = re.findall(r"[-+]?\d*\.?\d+", text)
    rri = np.array([float(t) for t in tokens if float(t) > 0])

    if len(rri) == 0:
        raise ValueError("No numeric values found in the uploaded file.")

    # Auto-detect unit
    if rri.mean() < 3.0:          # seconds
        rri = rri * 1000.0
    elif rri.mean() > 2000.0:     # already in samples at some rate? unlikely but clamp
        pass

    # Physiological sanity clamp: 300–2000 ms
    rri = rri[(rri >= 300) & (rri <= 2000)]
    if len(rri) < 30:
        raise ValueError(
            f"Too few valid RR intervals ({len(rri)}). "
            "Need at least 30 beats per phase."
        )
    return rri


def _rr_to_peaks(rri_ms: np.ndarray) -> np.ndarray:
    """Convert RR intervals (ms) to R-peak sample indices at 1000 Hz."""
    peaks = np.concatenate([[0], np.cumsum(rri_ms[:-1])]).astype(int)
    return peaks


def _hrv_one_phase(rri_ms: np.ndarray) -> dict:
    """
    Compute the 21 HRV metrics for one phase from RR intervals in ms.
    Returns a dict of raw (un-transformed) values keyed by base feature name.
    Missing / failed metrics are NaN.
    """
    out = {}

    if not _NK_OK or len(rri_ms) < 15:
        # Return NaN dict — scaler will use training mean (0 in scaled space)
        keys = [
            "RMSSD_ms","SDNN_ms","pNNxx_pct","Mean_RR_ms","Mean_HR_bpm",
            "RR_tri_index","TINN_ms",
            "LF_abs_FFT","HF_abs_FFT","LF_HF_ratio_FFT","VLF_abs_FFT",
            "Total_power_FFT","LF_nu_FFT","HF_nu_FFT",
            "SD1_ms","SD2_ms","SD2_SD1_ratio","SampEn","DFA_alpha1",
            "RQA_RecurrenceRate","RQA_Determinism",
        ]
        return {k: np.nan for k in keys}

    peaks = _rr_to_peaks(rri_ms)

    # ── Time + frequency + nonlinear in one call ──────────────────────────────
    try:
        hrv = nk.hrv(peaks, sampling_rate=1000, show=False)
    except Exception:
        hrv = pd.DataFrame()

    def g(col, default=np.nan):
        return float(hrv[col].iloc[0]) if col in hrv.columns else default

    out["RMSSD_ms"]         = g("HRV_RMSSD")
    out["SDNN_ms"]          = g("HRV_SDNN")
    out["pNNxx_pct"]        = g("HRV_pNN50")
    out["Mean_RR_ms"]       = g("HRV_MeanNN")
    mean_nn = out["Mean_RR_ms"]
    out["Mean_HR_bpm"]      = (60000.0 / mean_nn) if (mean_nn and mean_nn > 0) else np.nan
    out["RR_tri_index"]     = g("HRV_HTI")
    out["TINN_ms"]          = g("HRV_TINN")
    out["LF_abs_FFT"]       = g("HRV_LF")
    out["HF_abs_FFT"]       = g("HRV_HF")
    out["LF_HF_ratio_FFT"]  = g("HRV_LFHF")
    out["VLF_abs_FFT"]      = g("HRV_VLF")
    out["Total_power_FFT"]  = g("HRV_TP")
    out["LF_nu_FFT"]        = g("HRV_LFn")
    out["HF_nu_FFT"]        = g("HRV_HFn")
    out["SD1_ms"]           = g("HRV_SD1")
    out["SD2_ms"]           = g("HRV_SD2")
    # neurokit2 SD1SD2 = SD1/SD2; study uses SD2/SD1
    sd1sd2 = g("HRV_SD1SD2")
    out["SD2_SD1_ratio"]    = (1.0 / sd1sd2) if (sd1sd2 and sd1sd2 != 0) else np.nan
    out["SampEn"]           = g("HRV_SampEn")
    out["DFA_alpha1"]       = g("HRV_DFA_alpha1")

    # ── RQA ──────────────────────────────────────────────────────────────────
    try:
        rqa, _ = nk.complexity_rqa(rri_ms.astype(float), delay=1, dimension=2,
                                   show=False)
        out["RQA_RecurrenceRate"] = float(rqa["RecurrenceRate"].iloc[0])
        out["RQA_Determinism"]    = float(rqa["Determinism"].iloc[0])
    except Exception:
        out["RQA_RecurrenceRate"] = np.nan
        out["RQA_Determinism"]    = np.nan

    return out


def compute_hrv_features(rri_ms: np.ndarray, rest_beats: int) -> dict:
    """
    Split RR series into REST / EXERCISE phases and compute all 44 named features.

    Parameters
    ----------
    rri_ms     : full RR series in milliseconds
    rest_beats : number of beats belonging to the REST phase (from the start)

    Returns
    -------
    dict mapping final feature name (matching pca_hr_features.json) → value
    """
    rest_beats = max(15, min(rest_beats, len(rri_ms) - 15))
    rri_rest = rri_ms[:rest_beats]
    rri_ex   = rri_ms[rest_beats:]

    rest = _hrv_one_phase(rri_rest)
    ex   = _hrv_one_phase(rri_ex)

    # GLOBAL = metrics on the complete recording
    rri_global = rri_ms
    peaks_g    = _rr_to_peaks(rri_global)
    try:
        hrv_g = nk.hrv(peaks_g, sampling_rate=1000, show=False)
        def gg(c): return float(hrv_g[c].iloc[0]) if c in hrv_g.columns else np.nan
    except Exception:
        def gg(c): return np.nan

    g_ex_rmssd   = gg("HRV_RMSSD")   # whole-session RMSSD (exercise-dominated)
    g_ex_sdnn    = gg("HRV_SDNN")
    g_rest_rmssd = float(rest["RMSSD_ms"])
    g_rest_sdnn  = float(rest["SDNN_ms"])

    # nabla_h: HR reactivity and recovery ratios
    r_rmssd  = rest["RMSSD_ms"]
    e_rmssd  = ex["RMSSD_ms"]
    nabla_ex   = (e_rmssd / r_rmssd) if (r_rmssd and r_rmssd > 0) else np.nan
    nabla_rest = (r_rmssd / e_rmssd) if (e_rmssd and e_rmssd > 0) else np.nan
    g_nabla_ex   = (g_ex_rmssd / g_rest_rmssd) if (g_rest_rmssd and g_rest_rmssd > 0) else np.nan
    g_nabla_rest = (g_rest_rmssd / g_ex_rmssd)  if (g_ex_rmssd  and g_ex_rmssd  > 0) else np.nan

    # Build final dict with exact base names (log transforms applied in pca_transform.py)
    feats = {}
    for key, val in ex.items():
        feats[f"EX_{key}"]   = val
    for key, val in rest.items():
        feats[f"REST_{key}"]  = val

    feats["nabla_h_ex"]             = nabla_ex
    feats["nabla_h_rest"]           = nabla_rest
    feats["GLOBAL_EX_RMSSD_ms"]     = g_ex_rmssd
    feats["GLOBAL_EX_SDNN_ms"]      = g_ex_sdnn
    feats["GLOBAL_REST_RMSSD_ms"]   = g_rest_rmssd
    feats["GLOBAL_REST_SDNN_ms"]    = g_rest_sdnn
    feats["GLOBAL_nabla_h_ex"]      = g_nabla_ex
    feats["GLOBAL_nabla_h_rest"]    = g_nabla_rest

    return feats
