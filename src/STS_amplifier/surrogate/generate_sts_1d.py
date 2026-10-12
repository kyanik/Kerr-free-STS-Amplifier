from __future__ import annotations

from pathlib import Path
import numpy as np

from ..core import parametric_metrics
from ..params import STS_params


def generate_sts_1d_dataset(
    *,
    out_npz: str,
    nsamples: int = 50,
    n_levels: int = 50,
    # fixed sim constants
    kappa: float = 300e6 * 2 * np.pi,
    drive: float = 0.2,
    theta: float = -np.pi / 2,
    phi: float = 0.0,
    # fixed (as requested)
    Delta: float = 0.0,
    gamma: float = 0.0,
    # STS physical parameters (required)
    STS_EC: float,
    STS_EL: float,
    # sampling range for lam_norm = lam/(kappa/4)
    lam_norm_min: float = -1.93,
    lam_norm_max: float = 0.0,
    # sampling control
    sample_mode: str = "grid",   # "grid" or "random"
    seed: int = 0,
) -> None:
    """
    STS-only 1D surrogate dataset (gamma=0, Delta=0 by default).

    Feature:
      X[i,0] = lam_norm = lam / (kappa/4)

    Targets:
      y[i,0] = G_dB
      y[i,1] = eta

    Saved to out_npz as npz with arrays:
      X, y, lam_norm, feature_names, target_names, constants
    """
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ratio, _phi_zps = STS_params(STS_EC, STS_EL)

    if nsamples < 2 and sample_mode == "grid":
        raise ValueError("nsamples must be >= 2 for sample_mode='grid'")

    if sample_mode == "grid":
        lam_norm = np.linspace(lam_norm_min, lam_norm_max, nsamples)
    elif sample_mode == "random":
        rng = np.random.default_rng(seed)
        lam_norm = rng.uniform(lam_norm_min, lam_norm_max, size=nsamples)
    else:
        raise ValueError("sample_mode must be 'grid' or 'random'")

    lam = lam_norm * (kappa / 4.0)

    X = lam_norm.reshape(-1, 1).astype(float)
    y = np.zeros((nsamples, 2), dtype=float)

    for i in range(nsamples):
        _g, _G_lin, G_dB, _F_norm, eta = parametric_metrics(
            "STS",
            Delta=Delta,
            kappa=kappa,
            gamma=gamma,
            lam=lam[i],
            drive=drive,
            n_levels=n_levels,
            theta=theta,
            phi=phi,
            Kerr=0.0,
            ratio=ratio,
        )

        y[i, 0] = float(np.real(G_dB))
        y[i, 1] = float(np.real(eta))

        if (i + 1) % max(1, nsamples // 10) == 0:
            print(f"generated {i+1}/{nsamples}")

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        lam_norm=lam_norm,
        feature_names=np.array(["lam_norm"], dtype=object),
        target_names=np.array(["G_dB", "eta"], dtype=object),
        constants=np.array(
            [
                "kind=STS",
                f"Delta={Delta}",
                f"gamma={gamma}",
                f"drive={drive}",
                f"kappa={kappa}",
                f"theta={theta}",
                f"phi={phi}",
                f"n_levels={n_levels}",
                f"STS_EC={STS_EC}",
                f"STS_EL={STS_EL}",
                f"lam_norm_range=[{lam_norm_min},{lam_norm_max}]",
                f"sample_mode={sample_mode}",
                f"seed={seed}",
            ],
            dtype=object,
        ),
    )
    print(f"saved dataset: {out_npz}")