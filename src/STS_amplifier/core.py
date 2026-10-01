from __future__ import annotations

import cmath
import numpy as np
import qutip as qt
from functools import lru_cache

from .params import STS_params, jpa_params


def _quadratures_from_a(a_c):
    X = (a_c + np.conj(a_c)) / np.sqrt(2)
    P = 1j * (-a_c + np.conj(a_c)) / np.sqrt(2)
    return X, P


def _gain_matrix_from_two_probes(a_out1, a_in1, a_out2, a_in2):
    Xo1, Po1 = _quadratures_from_a(a_out1)
    Xi1, Pi1 = _quadratures_from_a(a_in1)

    Xo2, Po2 = _quadratures_from_a(a_out2)
    Xi2, Pi2 = _quadratures_from_a(a_in2)

    IN = np.array([[Xi1, Xi2], [Pi1, Pi2]], dtype=complex)
    OUT = np.array([[Xo1, Xo2], [Po1, Po2]], dtype=complex)

    return OUT @ np.linalg.inv(IN)


def _phase_preserving_gain_from_g(g):
    return 0.25 * np.abs(g[0, 0] + g[1, 1] + 1j * (g[1, 0] - g[0, 1])) ** 2


def _solve_two_probes_and_get_g(H0, a, kappa, drive, phi, c_ops):
    """
    Two steady-state solves:
      probe1: epsilon
      probe2: i*epsilon
    """
    def solve_for_epsilon(eps):
        a_in = 1j * eps / np.sqrt(kappa)  # bare kappa
        H_probe = eps * a.dag() + np.conj(eps) * a
        rho_ss = qt.steadystate(H0 + H_probe, c_ops)
        a_exp = qt.expect(a, rho_ss)
        a_out = np.sqrt(kappa) * a_exp + a_in  # bare kappa
        return a_out, a_in

    eps1 = drive * kappa * np.exp(-1j * phi)
    eps2 = drive * kappa * 1j * np.exp(-1j * phi)

    a_out1, a_in1 = solve_for_epsilon(eps1)
    a_out2, a_in2 = solve_for_epsilon(eps2)

    g = _gain_matrix_from_two_probes(a_out1, a_in1, a_out2, a_in2)
    G_lin = _phase_preserving_gain_from_g(g)
    return g, G_lin, eps1


def _vacuum_variance(op, n_levels):
    vac = qt.basis(n_levels, 0)
    exp1 = 0.5 * qt.expect(op.dag() * op + op * op.dag(), vac)
    exp2 = np.abs(qt.expect(op, vac)) ** 2
    return exp1 - exp2


def extract_Kerr(kind, params):
    if kind.upper() == "DPA":
        ratio, Kerr, zpf = 0.0, 0.0, 0.0

    elif kind.upper() == "JPA":
        if len(params) != 3:
            raise ValueError("For kind='JPA', params must be (F_JPA, EC_J, EJ).")
        ratio, Kerr, zpf = jpa_params(*params)

    elif kind.upper() == "STS":
        Kerr = 0.0
        if len(params) != 2:
            raise ValueError("For kind='STS', params must be (EC, EL).")
        ratio, zpf = STS_params(*params)

    else:
        raise ValueError("kind must be one of: 'DPA', 'JPA', 'STS'")

    return ratio, Kerr, zpf


def parametric_metrics(
    kind,
    Delta, kappa, gamma, lam, drive, n_levels, theta, phi,
    Kerr=0.0, ratio=0.0,
):

    a = qt.destroy(n_levels)

    # gamma ONLY here:
    kappa_eff = kappa + gamma
    c_ops = [np.sqrt(kappa_eff) * a]

    # Quadratic DPA Hamiltonian
    lam_phase = lam * np.exp(-1j * theta)
    H_DPA = (
        Delta * a.dag() * a
        + (lam_phase / 2) * a.dag() * a.dag()
        + (np.conj(lam_phase) / 2) * a * a
    )

    # Nonlinear terms
    H_nl = 0
    if kind.upper() == "DPA":
        LAM = 0.0

    elif kind.upper() == "JPA":
        LAM = -ratio * lam
        H_nl = (
            Kerr * a.dag() * a.dag() * a * a
            + LAM * (a.dag() * a.dag() * a.dag() * a + a.dag() * a * a * a)
        )

    elif kind.upper() == "STS":
        LAM = -ratio * lam
        H_nl = LAM * (a.dag() * a.dag() * a.dag() * a + a.dag() * a * a * a)

    else:
        raise ValueError("kind must be one of: 'DPA', 'JPA', 'STS'")

    H0 = H_DPA + H_nl

    # Gain (linear)
    g, G_lin, epsilon = _solve_two_probes_and_get_g(H0, a, kappa, drive, phi, c_ops)

    # Convenience: dB gain (DO NOT use for physics formulas)
    G_dB = 10.0 * np.log10(G_lin) if np.real(G_lin) > 0 else np.nan

    # Coefficient normalization (use linear gain)
    # NOTE: your notebook comment: missing detuning Delta terms
    if kind.upper() == "DPA":
        Coeff = kappa / 4 + (lam**2) / kappa
    else:
        gsw = cmath.sqrt(G_lin)
        lam_DPA = 1j * kappa * cmath.sqrt(-1 + gsw) / cmath.sqrt(-4 - 4 * gsw)
        Coeff = kappa / 4 + (np.abs(lam_DPA) ** 2) / kappa

    # ain_dag (gamma does NOT enter)
    base = (
        ((-1j * Delta + kappa / 2) * a.dag() - 1j * np.conj(lam_phase) * a) / np.sqrt(kappa)
        - 1j * np.conj(epsilon) / np.sqrt(kappa)
    )

    if kind.upper() == "DPA":
        ain_dag = base

    elif kind.upper() == "JPA":
        ain_dag = (
            base
            - 1j * 2 * Kerr * a.dag() * a.dag() * a / np.sqrt(kappa)
            - 1j * LAM * (a.dag() * a.dag() * a.dag() + 3 * a.dag() * a * a) / np.sqrt(kappa)
        )

    elif kind.upper() == "STS":
        ain_dag = (
            base
            + 1j * LAM * (a.dag() * a.dag() * a.dag() + 3 * a.dag() * a * a) / np.sqrt(kappa)
        )

    var_aindag = _vacuum_variance(ain_dag, n_levels)

    F = (G_lin - 1) * var_aindag / G_lin
    F_norm = F / Coeff
    eta = 1 / (1 + 2 * F_norm)

    return g, G_lin, G_dB, F_norm, eta


def _lam_sweep(kappa, span_factor, npts):
    return -np.linspace(0.0, span_factor, npts) * (kappa / 4.0)


def _lam_conditions(kind, params, kappa, npts):
    if kind.upper() == "DPA":
        lam_max = 1.93
        return _lam_sweep(kappa, lam_max, npts)

    elif kind.upper() == "JPA":
        ratio, Kerr, _zpf = extract_Kerr("JPA", params)
        if np.abs(Kerr / kappa) <= 0.01:
            lam_max = 1.93
            return _lam_sweep(kappa, lam_max, npts)
        lam_max1 = 1.93
        lam_max2 = 3.93
        return _lam_sweep(kappa, lam_max1, npts), _lam_sweep(kappa, lam_max2, npts)

    elif kind.upper() == "STS":
        lam_max = 1.93
        return _lam_sweep(kappa, lam_max, npts)

    else:
        raise ValueError("kind must be one of: 'DPA', 'JPA', 'STS'")


def _sweep_G_eta(kind, params, npts, Delta, kappa, gamma, drive, n_levels, theta, phi):
    ratio, Kerr, _zpf = extract_Kerr(kind, params)

    if kind.upper() == "JPA" and np.abs(Kerr / kappa) > 0.01:
        lamsG, lams_eta = _lam_conditions(kind, params, kappa, npts)

        G_g = np.full(len(lamsG), np.nan, dtype=float)
        G_eta = np.full(len(lams_eta), np.nan, dtype=float)
        eta = np.full(len(lams_eta), np.nan, dtype=float)

        for i, lam in enumerate(lamsG):
            _, _, G_g[i], _, _ = parametric_metrics(
                kind, Delta, kappa, gamma, lam, drive, n_levels, theta, phi, Kerr, ratio
            )
        for i, lam in enumerate(lams_eta):
            _, _, G_eta[i], _, eta[i] = parametric_metrics(
                kind, Delta, kappa, gamma, lam, drive, n_levels, theta, phi, Kerr, ratio
            )
        return G_g, G_eta, eta

    lams = _lam_conditions(kind, params, kappa, npts)
    G = np.full(len(lams), np.nan, dtype=float)
    eta = np.full(len(lams), np.nan, dtype=float)
    for i, lam in enumerate(lams):
        _, _, G[i], _, eta[i] = parametric_metrics(
            kind, Delta, kappa, gamma, lam, drive, n_levels, theta, phi, Kerr, ratio
        )
    return G, G, eta