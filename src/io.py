"""
io.py - Load, clean, and slice DMA data from the Henkel EP5089 Excel file.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Column name constants (match the Excel headers) ─────────────────────────
COL_FREQ   = "Freq (rad/sec)"          # angular frequency [rad/s]
COL_TEMP_C = "Temp C"                   # temperature [°C]  (numeric)
COL_GP     = "Storage Modulus G' (Mpa)" # G' [MPa]
COL_GPP    = 'Loss Modulus G"(Mpa)'     # G'' [MPa]
COL_TAND   = "Loss Factor (Tan δ)"      # tan δ  [-]

RAW_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "Henkel EP5089-DMTA Data.xlsx"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Excel sheet and return a tidy DataFrame.

    Returned columns (renamed for convenience):
        omega_rad_s, temp_C, Gp_MPa, Gpp_MPa, tan_delta
    """
    filepath = Path(filepath) if filepath else RAW_FILE
    df = pd.read_excel(filepath, sheet_name="Worksheet")

    # Keep only what we need and rename
    df = df.rename(columns={
        COL_FREQ:   "omega_rad_s",
        COL_TEMP_C: "temp_C",
        COL_GP:     "Gp_MPa",
        COL_GPP:    "Gpp_MPa",
        COL_TAND:   "tan_delta",
    })[[
        "omega_rad_s", "temp_C", "Gp_MPa", "Gpp_MPa", "tan_delta"
    ]]

    df = df.sort_values(["temp_C", "omega_rad_s"]).reset_index(drop=True)
    return df


def available_temperatures(df: pd.DataFrame) -> np.ndarray:
    """Return sorted array of unique temperatures [°C] in the dataset."""
    return np.sort(df["temp_C"].unique())


def select_temperature(df: pd.DataFrame, temp_C: float) -> pd.DataFrame:
    """Extract a single-temperature slice, sorted by ascending ω."""
    mask = df["temp_C"] == temp_C
    if mask.sum() == 0:
        avail = available_temperatures(df)
        raise ValueError(
            f"Temperature {temp_C} °C not found. Available: {avail}"
        )
    return df.loc[mask].sort_values("omega_rad_s").reset_index(drop=True)


def save_processed(df: pd.DataFrame, tag: str = "slice") -> Path:
    """Save a processed DataFrame to CSV in data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{tag}.csv"
    df.to_csv(out, index=False)
    return out
