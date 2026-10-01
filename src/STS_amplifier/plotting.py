import os
import matplotlib.pyplot as plt

from .core import _sweep_G_eta


def plot_gain_eta_from_physical_params(
    *,
    Delta, kappa, gamma, drive, n_levels, theta, phi, npts,
    # STS physical params
    STS_EC, STS_EL,
    # JPA physical params (three JPAs)
    JPA1_F, JPA1_EC, JPA1_EJ,
    JPA2_F, JPA2_EC, JPA2_EJ,
    JPA3_F, JPA3_EC, JPA3_EJ,
    figsize=(12, 5),
    save_path: str | None = None,   # NEW
    dpi: int = 200,                 # NEW
):
    # These “params tuples” are now constructed from user input
    parDPA = (0.0, 0.0, 0.0)  # ignored by DPA path
    parSTS = (STS_EC, STS_EL)
    parJPA1 = (JPA1_F, JPA1_EC, JPA1_EJ)
    parJPA2 = (JPA2_F, JPA2_EC, JPA2_EJ)
    parJPA3 = (JPA3_F, JPA3_EC, JPA3_EJ)

    gain_curves = []
    eta_curves = []

    if gamma != 0.0:
        G_x, _, _ = _sweep_G_eta("DPA", parDPA, npts, Delta, kappa, 0.0, drive, n_levels, theta, phi)

    G_DPA, _, eta_DPA = _sweep_G_eta("DPA", parDPA, npts, Delta, kappa, gamma, drive, n_levels, theta, phi)
    gain_curves.append(("DPA", G_DPA))
    eta_curves.append(("DPA", G_DPA, eta_DPA))

    G_STS, _, eta_STS = _sweep_G_eta("STS", parSTS, npts, Delta, kappa, gamma, drive, n_levels, theta, phi)
    gain_curves.append(("STS", G_STS))
    eta_curves.append(("STS", G_STS, eta_STS))

    Gg_JPA1, G_eta_JPA1, eta_JPA1 = _sweep_G_eta("JPA", parJPA1, npts, Delta, kappa, gamma, drive, n_levels, theta, phi)
    gain_curves.append(("JPA1", Gg_JPA1))
    eta_curves.append(("JPA1", G_eta_JPA1, eta_JPA1))

    Gg_JPA2, G_eta_JPA2, eta_JPA2 = _sweep_G_eta("JPA", parJPA2, npts, Delta, kappa, gamma, drive, n_levels, theta, phi)
    gain_curves.append(("JPA2", Gg_JPA2))
    eta_curves.append(("JPA2", G_eta_JPA2, eta_JPA2))

    Gg_JPA3, G_eta_JPA3, eta_JPA3 = _sweep_G_eta("JPA", parJPA3, npts, Delta, kappa, gamma, drive, n_levels, theta, phi)
    gain_curves.append(("JPA3", Gg_JPA3))
    eta_curves.append(("JPA3", G_eta_JPA3, eta_JPA3))

    # Plotting
    if gamma != 0.0:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = None
        x = G_x
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        x = G_DPA

    for label, y in gain_curves:
        ax1.plot(x, y, lw=2, label=label)

    ax1.set_xlabel(r"DPA phase-preserving gain $G_{\mathrm{DPA}}$ (dB), $\gamma=0$")
    ax1.set_ylabel(r"Design phase-preserving gain $G$ (dB)")
    ax1.set_title(fr"Gain comparison ($\gamma/\kappa$ = {(gamma/kappa):.2g})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if ax2 is not None:
        for label, G, eta in eta_curves:
            ax2.plot(G, eta, lw=2, label=label)

        ax2.set_xlabel(r"Phase-preserving gain $G$ (dB)")
        ax2.set_ylabel(r"Quantum efficiency $\eta$")
        ax2.set_title(r"Efficiency vs Gain $\gamma=0$")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(0.0, 25.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()

    # NEW: optional save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig