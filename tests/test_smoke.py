from STS_amplifier.core import parametric_metrics


def test_parametric_metrics_smoke():
    # small Hilbert space for speed
    g, G_lin, G_dB, F_norm, eta = parametric_metrics(
        "DPA",
        Delta=0.0,
        kappa=1.0,
        gamma=0.0,
        lam=-0.1,
        drive=0.01,
        n_levels=10,
        theta=0.0,
        phi=0.0,
        Kerr=0.0,
        ratio=0.0,
    )
    assert G_lin == G_lin  # not NaN