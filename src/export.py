"""
export.py - Export fitted Prony parameters to CSV / JSON for FE solvers.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLE_DIR   = RESULTS_DIR / "tables"


def prony_table(g: np.ndarray, tau: np.ndarray, G0: float, G_eq: float) -> pd.DataFrame:
    """Build a human-readable Prony parameter table."""
    df = pd.DataFrame({
        "i":        np.arange(1, len(g) + 1),
        "g_i":      g,
        "tau_i_s":  tau,
        "G_i_MPa":  G0 * g,
    })
    # Append a summary row
    summary = pd.DataFrame([{
        "i":       "sum",
        "g_i":     g.sum(),
        "tau_i_s": np.nan,
        "G_i_MPa": (G0 * g).sum(),
    }])
    df = pd.concat([df, summary], ignore_index=True)
    return df


def export_csv(g, tau, G0, G_eq, filename: str = "prony_params.csv") -> Path:
    """Save Prony parameters to CSV."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / filename
    df = prony_table(g, tau, G0, G_eq)
    df.to_csv(out, index=False)
    return out


def export_metrics_json(metrics: dict, filename: str = "fit_metrics.json") -> Path:
    """Save fit-quality metrics to JSON."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / filename
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out


def export_abaqus_snippet(g, tau, G0, G_eq) -> str:
    """Return an Abaqus-style *VISCOELASTIC keyword block (Prony form)."""
    lines = ["*VISCOELASTIC, TIME=PRONY"]
    for gi, ti in zip(g, tau):
        # Abaqus expects:  g_i_shear, k_i_bulk (0 = incompressible), tau_i
        lines.append(f"  {gi:.8e},  0.0,  {ti:.8e}")
    return "\n".join(lines)
