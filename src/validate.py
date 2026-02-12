"""
validate.py - Plotting and error metrics for Prony fit validation.
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR     = RESULTS_DIR / "figures"

# ── Plotting style defaults ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       120,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "lines.linewidth":  1.8,
    "lines.markersize": 5,
})


# ── Error metrics ────────────────────────────────────────────────────────────

def rmse(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_error_pct(y_true: NDArray, y_pred: NDArray) -> float:
    """Mean absolute relative error in percent."""
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-12)))) * 100.0


def compute_metrics(omega, Gp_data, Gpp_data, Gp_pred, Gpp_pred) -> dict:
    """Return a dict of error metrics for G', G'', and tan δ."""
    tand_data = Gpp_data / np.maximum(Gp_data, 1e-12)
    tand_pred = Gpp_pred / np.maximum(Gp_pred, 1e-12)
    return {
        "Gp_RMSE_MPa":       rmse(Gp_data, Gp_pred),
        "Gp_relErr_pct":     relative_error_pct(Gp_data, Gp_pred),
        "Gpp_RMSE_MPa":      rmse(Gpp_data, Gpp_pred),
        "Gpp_relErr_pct":    relative_error_pct(Gpp_data, Gpp_pred),
        "tand_RMSE":         rmse(tand_data, tand_pred),
        "tand_relErr_pct":   relative_error_pct(tand_data, tand_pred),
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_Gp(omega, Gp_data, Gp_pred, title_suffix: str = "", save: bool = True):
    """Plot measured vs predicted G'(ω)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(omega, Gp_data,  "o", label="Measured G'", markerfacecolor="none")
    ax.semilogx(omega, Gp_pred,  "-", label="Prony fit G'")
    ax.set_xlabel("ω  [rad/s]")
    ax.set_ylabel("G'  [MPa]")
    ax.set_title(f"Storage Modulus G'  {title_suffix}")
    ax.legend()
    fig.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "Gprime_fit.png")
    return fig, ax


def plot_Gpp(omega, Gpp_data, Gpp_pred, title_suffix: str = "", save: bool = True):
    """Plot measured vs predicted G''(ω)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(omega, Gpp_data, "s", label='Measured G"', markerfacecolor="none")
    ax.semilogx(omega, Gpp_pred, "-", label='Prony fit G"')
    ax.set_xlabel("ω  [rad/s]")
    ax.set_ylabel('G"  [MPa]')
    ax.set_title(f'Loss Modulus G"  {title_suffix}')
    ax.legend()
    fig.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "Gdoubleprime_fit.png")
    return fig, ax


def plot_tan_delta(omega, Gp_data, Gpp_data, Gp_pred, Gpp_pred,
                   title_suffix: str = "", save: bool = True):
    """Plot measured vs predicted tan δ(ω)."""
    tand_data = Gpp_data / np.maximum(Gp_data, 1e-12)
    tand_pred = Gpp_pred / np.maximum(Gp_pred, 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(omega, tand_data, "^", label="Measured tan δ", markerfacecolor="none")
    ax.semilogx(omega, tand_pred, "-", label="Prony fit tan δ")
    ax.set_xlabel("ω  [rad/s]")
    ax.set_ylabel("tan δ  [-]")
    ax.set_title(f"Loss Factor  tan δ  {title_suffix}")
    ax.legend()
    fig.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "tandelta_fit.png")
    return fig, ax


def plot_overview(omega, Gp_data, Gpp_data, Gp_pred, Gpp_pred,
                  title_suffix: str = "", save: bool = True):
    """3-panel overview: G', G'', tan δ side by side."""
    tand_data = Gpp_data / np.maximum(Gp_data, 1e-12)
    tand_pred = Gpp_pred / np.maximum(Gp_pred, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")

    # G'
    axes[0].semilogx(omega, Gp_data,  "o", label="Data", markerfacecolor="none")
    axes[0].semilogx(omega, Gp_pred,  "-", label="Fit")
    axes[0].set_xlabel("ω [rad/s]"); axes[0].set_ylabel("G' [MPa]")
    axes[0].set_title("G'"); axes[0].legend()

    # G''
    axes[1].semilogx(omega, Gpp_data, "s", label="Data", markerfacecolor="none")
    axes[1].semilogx(omega, Gpp_pred, "-", label="Fit")
    axes[1].set_xlabel("ω [rad/s]"); axes[1].set_ylabel('G" [MPa]')
    axes[1].set_title('G"'); axes[1].legend()

    # tan δ
    axes[2].semilogx(omega, tand_data, "^", label="Data", markerfacecolor="none")
    axes[2].semilogx(omega, tand_pred, "-", label="Fit")
    axes[2].set_xlabel("ω [rad/s]"); axes[2].set_ylabel("tan δ [-]")
    axes[2].set_title("tan δ"); axes[2].legend()

    fig.suptitle(f"Prony Fit Overview  {title_suffix}", fontsize=14)
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "overview_fit.png", bbox_inches="tight")
    return fig, axes
