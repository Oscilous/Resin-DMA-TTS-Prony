"""
tts.py - Time-Temperature Superposition (TTS) master curve construction.

Shifts individual temperature-frequency DMA sweeps onto a single reduced-
frequency axis using horizontal shift factors aT. Fits Williams-Landel-Ferry
(WLF) and Arrhenius models to the shift factors.
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import curve_fit

from .io import load_raw, available_temperatures, RAW_FILE


# ── Shift-factor extraction from the raw Excel file ─────────────────────────

def load_shift_factors(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load the dataset-provided shift factors (one per temperature).

    Returns a DataFrame with columns: temp_C, aT, log_aT
    sorted by ascending temperature.
    """
    filepath = Path(filepath) if filepath else RAW_FILE
    raw = pd.read_excel(filepath, sheet_name="Worksheet")

    sf = (
        raw.groupby("Temp C")
        .agg(aT=("Shift Factors", "first"),
             log_aT=("Log (Shift factors)", "first"))
        .reset_index()
        .rename(columns={"Temp C": "temp_C"})
        .sort_values("temp_C")
        .reset_index(drop=True)
    )
    return sf


# ── Build master curve ───────────────────────────────────────────────────────

def build_master_curve(
    df_all: pd.DataFrame,
    sf: pd.DataFrame,
    T_ref: float | None = None,
) -> pd.DataFrame:
    """Shift all temperature slices to build a master curve.

    Parameters
    ----------
    df_all : Full DMA DataFrame (output of io.load_raw)
    sf     : Shift-factor table (output of load_shift_factors)
    T_ref  : Reference temperature [°C].  If None, auto-detected as the
             temperature where |log aT| is smallest (≈ 0).

    Returns
    -------
    mc : DataFrame with columns
         omega_rad_s, omega_reduced, temp_C, Gp_MPa, Gpp_MPa, tan_delta, aT
         sorted by omega_reduced.
    """
    if T_ref is None:
        T_ref = float(sf.loc[sf["log_aT"].abs().idxmin(), "temp_C"])

    # Merge shift factors into the data
    mc = df_all.merge(sf[["temp_C", "aT"]], on="temp_C", how="left")

    # Reduced frequency: ω_red = aT × ω
    mc["omega_reduced"] = mc["aT"] * mc["omega_rad_s"]
    mc = mc.sort_values("omega_reduced").reset_index(drop=True)

    return mc, T_ref


# ── WLF model ────────────────────────────────────────────────────────────────

def wlf(T: NDArray, C1: float, C2: float, T_ref: float) -> NDArray:
    """WLF equation: log10(aT) = -C1 (T - T_ref) / (C2 + T - T_ref)

    Singularity at T = T_ref - C2 is guarded (returns ±large value).
    """
    T = np.asarray(T, dtype=float)
    denom = C2 + (T - T_ref)
    # Guard against singularity (np.sign(0)==0 would fail)
    sign = np.where(denom >= 0, 1.0, -1.0)
    denom = np.where(np.abs(denom) < 1e-10, sign * 1e-10, denom)
    return -C1 * (T - T_ref) / denom


def fit_wlf(
    temp_C: NDArray,
    log_aT: NDArray,
    T_ref: float,
    *,
    T_range: tuple[float, float] | None = None,
) -> dict:
    """Fit the WLF equation to experimental shift factors.

    Parameters
    ----------
    temp_C, log_aT : Experimental data arrays.
    T_ref : Reference temperature [°C].
    T_range : Optional (T_lo, T_hi) to restrict the fit window.
              WLF is physically valid only near Tg, typically
              Tg-20 to Tg+100 C. If None the full range is used,
              but the singularity T_ref-C2 must stay outside the data.

    Returns dict with C1, C2, T_ref, log_aT_pred (full range), residual_std.
    """
    temp_C = np.asarray(temp_C, dtype=float)
    log_aT = np.asarray(log_aT, dtype=float)

    # --- Select fit window ---------------------------------------------------
    if T_range is not None:
        in_range = (temp_C >= T_range[0]) & (temp_C <= T_range[1])
    else:
        in_range = np.ones(len(temp_C), dtype=bool)

    # Remove the reference point (log_aT ≈ 0)
    mask = in_range & (np.abs(temp_C - T_ref) > 0.5)
    T_fit = temp_C[mask]
    y_fit = log_aT[mask]

    if len(T_fit) < 2:
        raise ValueError("Not enough data points in T_range for WLF fit.")

    def _wlf_fixed_Tref(T, C1, C2):
        return wlf(T, C1, C2, T_ref)

    # Bounds: keep singularity (T_ref - C2) at least 2 °C below the lowest
    # *fit* temperature so the denominator stays well-behaved.
    T_fit_min = float(T_fit.min())
    C2_min_for_safety = T_ref - T_fit_min + 2.0   # → T_sing < T_fit_min - 2
    C2_min_for_safety = max(C2_min_for_safety, 5.0)

    # Initial guess & bounds
    C2_init = C2_min_for_safety + 10.0             # a few degrees extra margin
    C2_upper = max(C2_min_for_safety + 200.0, 500.0)

    popt, pcov = curve_fit(
        _wlf_fixed_Tref, T_fit, y_fit,
        p0=[15.0, C2_init],
        bounds=([0.01, C2_min_for_safety], [300.0, C2_upper]),
        maxfev=20_000,
    )
    C1, C2 = popt

    # Predict on *all* temperatures (may hit singularity outside fit window)
    log_aT_pred = wlf(temp_C, C1, C2, T_ref)

    # Residuals only on fit window
    log_aT_fit_pred = wlf(T_fit, C1, C2, T_ref)
    residual = y_fit - log_aT_fit_pred
    valid = np.isfinite(residual)

    return {
        "C1": C1,
        "C2": C2,
        "T_ref": T_ref,
        "T_range_used": (float(T_fit.min()), float(T_fit.max())),
        "log_aT_pred": log_aT_pred,
        "residual_std": float(np.std(residual[valid])) if valid.any() else float("nan"),
    }


# ── Arrhenius model ──────────────────────────────────────────────────────────

def arrhenius(T_C: NDArray, Ea_over_R: float, T_ref_C: float) -> NDArray:
    """Arrhenius equation: log10(aT) = (Ea/R) / ln(10) × (1/T - 1/T_ref)
    where T is in Kelvin.
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    T_ref_K = T_ref_C + 273.15
    return (Ea_over_R / np.log(10.0)) * (1.0 / T_K - 1.0 / T_ref_K)


def fit_arrhenius(temp_C: NDArray, log_aT: NDArray, T_ref: float) -> dict:
    """Fit Arrhenius model to shift factors.

    Returns dict with Ea_over_R [K], Ea [kJ/mol], and fitted curve.
    """
    temp_C = np.asarray(temp_C, dtype=float)
    log_aT = np.asarray(log_aT, dtype=float)

    mask = np.abs(temp_C - T_ref) > 0.5

    def _arr(T_C, Ea_over_R):
        return arrhenius(T_C, Ea_over_R, T_ref)

    popt, pcov = curve_fit(_arr, temp_C[mask], log_aT[mask], p0=[5000.0])
    Ea_over_R = popt[0]

    R = 8.314  # J/(mol·K)
    Ea_kJ_mol = Ea_over_R * R / 1000.0

    log_aT_pred = arrhenius(temp_C, Ea_over_R, T_ref)

    return {
        "Ea_over_R_K": Ea_over_R,
        "Ea_kJ_mol": Ea_kJ_mol,
        "T_ref": T_ref,
        "log_aT_pred": log_aT_pred,
        "residual_std": float(np.std(log_aT - log_aT_pred)),
    }


# ── Convenience: extract master-curve arrays for fitting ─────────────────────

def master_curve_arrays(mc: pd.DataFrame):
    """Return sorted (omega_reduced, Gp, Gpp, tan_delta) numpy arrays."""
    mc_sorted = mc.sort_values("omega_reduced")
    return (
        mc_sorted["omega_reduced"].values,
        mc_sorted["Gp_MPa"].values,
        mc_sorted["Gpp_MPa"].values,
        mc_sorted["tan_delta"].values,
    )
