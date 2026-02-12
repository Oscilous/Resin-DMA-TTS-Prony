"""
fit.py - Least-squares Prony series fitting to DMA frequency-domain data.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .model import prony_Gp, prony_Gpp, unpack_params, pack_params


def _residuals(x: NDArray, omega: NDArray, Gp_data: NDArray, Gpp_data: NDArray,
               N: int, w_Gp: float, w_Gpp: float, use_log: bool = False) -> NDArray:
    """Combined weighted residuals for G' and G''."""
    g, tau, G0 = unpack_params(x, N)

    Gp_pred  = prony_Gp(omega, g, tau, G0)
    Gpp_pred = prony_Gpp(omega, g, tau, G0)

    if use_log:
        # Log-space residuals - better for broadband data spanning many decades.
        # Guard: mask out non-positive values
        eps = 1e-30
        res_Gp  = w_Gp  * (np.log10(np.maximum(Gp_pred, eps))
                          - np.log10(np.maximum(Gp_data, eps)))
        res_Gpp = w_Gpp * (np.log10(np.maximum(Gpp_pred, eps))
                          - np.log10(np.maximum(Gpp_data, eps)))
    else:
        # Relative residuals - good for single-temperature data
        res_Gp  = w_Gp  * (Gp_pred  - Gp_data)  / np.maximum(Gp_data,  1e-12)
        res_Gpp = w_Gpp * (Gpp_pred - Gpp_data) / np.maximum(Gpp_data, 1e-12)

    return np.concatenate([res_Gp, res_Gpp])


def initial_guess(omega: NDArray, Gp_data: NDArray, N: int) -> NDArray:
    """Generate a reasonable starting point.

    - G0 estimated as the maximum measured G'
    - τᵢ spaced log-uniformly across the inverse frequency range
    - gᵢ initialised to equal shares summing to ~0.5
    """
    G0_init = float(np.max(Gp_data)) * 1.05

    # Relaxation times spanning the inverse of the frequency range
    tau_min = 1.0 / omega.max()
    tau_max = 1.0 / omega.min()
    tau_init = np.logspace(np.log10(tau_min), np.log10(tau_max), N)

    g_init = np.full(N, 0.5 / N)

    return pack_params(g_init, tau_init, G0_init)


def fit_prony(
    omega: NDArray,
    Gp_data: NDArray,
    Gpp_data: NDArray,
    N: int = 5,
    w_Gp: float = 1.0,
    w_Gpp: float = 1.0,
    verbose: int = 1,
    use_log: bool | None = None,
) -> dict:
    """Fit a Prony series with *N* terms to frequency-domain DMA data.

    Parameters
    ----------
    omega    : angular frequency [rad/s]
    Gp_data  : measured storage modulus G'
    Gpp_data : measured loss modulus G''
    N        : number of Prony terms
    w_Gp     : weight on G' residuals
    w_Gpp    : weight on G'' residuals
    verbose  : 0 = silent, 1 = summary, 2 = iteration detail
    use_log  : If True, use log-space residuals (better for broadband data).
               If None (default), auto-detect: use log when frequency spans > 4 decades.

    Returns
    -------
    result : dict with keys
        g, tau, G0        - fitted Prony parameters
        G_eq              - equilibrium modulus
        Gp_pred, Gpp_pred - model predictions at the given omega
        opt               - scipy OptimizeResult object
    """
    omega   = np.asarray(omega, dtype=float)
    Gp_data = np.asarray(Gp_data, dtype=float)
    Gpp_data = np.asarray(Gpp_data, dtype=float)

    # Auto-detect broadband data
    if use_log is None:
        freq_decades = np.log10(omega.max() / omega.min())
        use_log = freq_decades > 4.0
    if use_log and verbose:
        print(f"Using log-space residuals (data spans "
              f"{np.log10(omega.max()/omega.min()):.1f} decades)")

    x0 = initial_guess(omega, Gp_data, N)

    # Bounds ------------------------------------------------------------------
    # g_i ∈ [0, 1],  log10(τ_i): adapt to data frequency range,  G0 ∈ [0, 10×max(G')]
    log_tau_lo = np.log10(1.0 / omega.max()) - 2.0   # 2 decades below fastest τ
    log_tau_hi = np.log10(1.0 / omega.min()) + 2.0   # 2 decades above slowest τ
    lb = np.concatenate([np.zeros(N), np.full(N, log_tau_lo), [0.0]])
    ub = np.concatenate([np.ones(N),  np.full(N, log_tau_hi), [10.0 * Gp_data.max()]])

    result = least_squares(
        _residuals, x0,
        args=(omega, Gp_data, Gpp_data, N, w_Gp, w_Gpp, use_log),
        bounds=(lb, ub),
        method="trf",
        max_nfev=100_000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        verbose=verbose,
    )

    g, tau, G0 = unpack_params(result.x, N)
    G_eq = G0 * (1.0 - np.sum(g))

    # Sort terms by ascending τ
    order = np.argsort(tau)
    g = g[order]
    tau = tau[order]

    return {
        "g":       g,
        "tau":     tau,
        "G0":      G0,
        "G_eq":    G_eq,
        "N":       N,
        "Gp_pred": prony_Gp(omega, g, tau, G0),
        "Gpp_pred": prony_Gpp(omega, g, tau, G0),
        "opt":     result,
    }


# ── Collocation method for broadband data ────────────────────────────────────

def _residuals_colloc(x: NDArray, omega: NDArray, Gp_data: NDArray,
                      Gpp_data: NDArray, tau_fixed: NDArray,
                      w_Gp: float, w_Gpp: float) -> NDArray:
    """Residuals with fixed tau - only g_i and G0 are free."""
    N = len(tau_fixed)
    g = x[:N]
    G0 = x[N]
    Gp_pred  = prony_Gp(omega, g, tau_fixed, G0)
    Gpp_pred = prony_Gpp(omega, g, tau_fixed, G0)
    # Log-space residuals for broadband data
    eps = 1e-30
    res_Gp  = w_Gp  * (np.log10(np.maximum(Gp_pred, eps))
                       - np.log10(np.maximum(Gp_data, eps)))
    res_Gpp = w_Gpp * (np.log10(np.maximum(Gpp_pred, eps))
                       - np.log10(np.maximum(Gpp_data, eps)))
    return np.concatenate([res_Gp, res_Gpp])


def fit_prony_collocation(
    omega: NDArray,
    Gp_data: NDArray,
    Gpp_data: NDArray,
    N: int = 20,
    w_Gp: float = 1.0,
    w_Gpp: float = 1.0,
    verbose: int = 1,
) -> dict:
    """Prony fit with **fixed** log-spaced τ_i (collocation method).

    Much more robust for broadband master-curve data because τ_i are
    fixed at one-per-decade, preventing the wild G'' oscillations that
    occur when τ positions are free to move.  Only the weights g_i and
    the glassy modulus G0 are optimised.

    N is automatically clamped to ≤ number of frequency decades.
    """
    omega    = np.asarray(omega, dtype=float)
    Gp_data  = np.asarray(Gp_data, dtype=float)
    Gpp_data = np.asarray(Gpp_data, dtype=float)

    # Fix τ at one per decade across the inverse-frequency range
    log_tau_lo = np.log10(1.0 / omega.max()) - 1.0
    log_tau_hi = np.log10(1.0 / omega.min()) + 1.0
    n_decades = log_tau_hi - log_tau_lo
    N = min(N, int(np.ceil(n_decades)))
    tau_fixed = np.logspace(log_tau_lo, log_tau_hi, N)

    if verbose:
        print(f"Collocation: {N} terms, τ range [{10**log_tau_lo:.1e}, "
              f"{10**log_tau_hi:.1e}] s, ~1 per {n_decades/N:.1f} decades")

    G0_init = float(Gp_data.max()) * 1.2
    g_init  = np.full(N, 0.5 / N)
    x0 = np.concatenate([g_init, [G0_init]])

    lb = np.concatenate([np.zeros(N), [0.0]])
    ub = np.concatenate([np.ones(N),  [10.0 * Gp_data.max()]])

    result = least_squares(
        _residuals_colloc, x0,
        args=(omega, Gp_data, Gpp_data, tau_fixed, w_Gp, w_Gpp),
        bounds=(lb, ub),
        method="trf",
        max_nfev=100_000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        verbose=verbose,
    )

    g  = result.x[:N]
    G0 = result.x[N]
    G_eq = G0 * (1.0 - np.sum(g))

    # Sort by ascending τ (already sorted, but just in case)
    order = np.argsort(tau_fixed)
    g = g[order]
    tau = tau_fixed[order]

    return {
        "g":       g,
        "tau":     tau,
        "G0":      G0,
        "G_eq":    G_eq,
        "N":       N,
        "Gp_pred": prony_Gp(omega, g, tau, G0),
        "Gpp_pred": prony_Gpp(omega, g, tau, G0),
        "opt":     result,
    }
