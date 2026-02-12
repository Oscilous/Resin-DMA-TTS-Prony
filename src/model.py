"""
model.py - Generalised Maxwell (Prony series) frequency-domain equations.

Prony series in the frequency domain:

    G'(w) = G_eq + G0 * sum(g_i * (w*tau_i)^2 / (1 + (w*tau_i)^2))
    G''(w) =       G0 * sum(g_i * (w*tau_i)   / (1 + (w*tau_i)^2))

where:
    G_eq  - long-time (equilibrium) modulus
    G0    - instantaneous (glassy) modulus
    g_i   - normalised Prony weight for term i  (sum(g_i) <= 1)
    tau_i - relaxation time for term i  [s]

Normalised formulation: G0 is the primary parameter.
"""

import numpy as np
from numpy.typing import NDArray


def prony_Gp(omega: NDArray, g: NDArray, tau: NDArray, G0: float) -> NDArray:
    """Storage modulus G'(ω) from Prony parameters.

    Parameters
    ----------
    omega : (M,) angular frequency [rad/s]
    g     : (N,) normalised Prony weights
    tau   : (N,) relaxation times [s]
    G0    : instantaneous (glassy) modulus [same unit as output]

    Returns
    -------
    Gp : (M,) storage modulus in same units as G0
    """
    omega = np.asarray(omega)
    g = np.asarray(g)
    tau = np.asarray(tau)

    # Equilibrium modulus
    G_eq = G0 * (1.0 - np.sum(g))

    # Summation over Prony terms
    wt = omega[:, None] * tau[None, :]          # (M, N)
    wt2 = wt ** 2
    Gp = G_eq + G0 * np.sum(g[None, :] * wt2 / (1.0 + wt2), axis=1)
    return Gp


def prony_Gpp(omega: NDArray, g: NDArray, tau: NDArray, G0: float) -> NDArray:
    """Loss modulus G''(ω) from Prony parameters.

    Parameters
    ----------
    omega : (M,) angular frequency [rad/s]
    g     : (N,) normalised Prony weights
    tau   : (N,) relaxation times [s]
    G0    : instantaneous (glassy) modulus

    Returns
    -------
    Gpp : (M,) loss modulus in same units as G0
    """
    omega = np.asarray(omega)
    g = np.asarray(g)
    tau = np.asarray(tau)

    wt = omega[:, None] * tau[None, :]
    wt2 = wt ** 2
    Gpp = G0 * np.sum(g[None, :] * wt / (1.0 + wt2), axis=1)
    return Gpp


def prony_tan_delta(omega: NDArray, g: NDArray, tau: NDArray, G0: float) -> NDArray:
    """Loss factor tan δ = G'' / G'."""
    Gp = prony_Gp(omega, g, tau, G0)
    Gpp = prony_Gpp(omega, g, tau, G0)
    return Gpp / Gp


def unpack_params(x: NDArray, N: int):
    """Unpack a flat parameter vector into (g, tau, G0).

    Layout: [g_1 .. g_N, log10(τ_1) .. log10(τ_N), G0]
    """
    g   = x[:N]
    tau = 10.0 ** x[N:2*N]   # stored as log10 for numerical conditioning
    G0  = x[2*N]
    return g, tau, G0


def pack_params(g: NDArray, tau: NDArray, G0: float) -> NDArray:
    """Pack Prony parameters into a flat vector."""
    return np.concatenate([g, np.log10(tau), [G0]])
