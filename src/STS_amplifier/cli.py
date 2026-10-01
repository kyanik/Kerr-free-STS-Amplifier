import argparse

from .plotting import plot_gain_eta_from_physical_params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="STS-amplifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    ge = sub.add_parser("gain-eta", help="Gain comparison + efficiency vs gain")

    # common sim params (these can have defaults)
    ge.add_argument("--Delta", type=float, default=0.0)
    ge.add_argument("--kappa", type=float, default=300e6 * 2 * 3.141592653589793)
    ge.add_argument("--gamma", type=float, default=0.0)
    ge.add_argument("--drive", type=float, default=0.2)
    ge.add_argument("--theta", type=float, default=-3.141592653589793 / 2)
    ge.add_argument("--phi", type=float, default=0.0)
    ge.add_argument("--n-levels", type=int, default=150)
    ge.add_argument("--npts", type=int, default=30)
    ge.add_argument("--figsize", type=float, nargs=2, default=(12, 5))

    # STS physical params (required)
    ge.add_argument("--STS-EC", type=float, required=True)
    ge.add_argument("--STS-EL", type=float, required=True)

    # JPA physical params (required)
    for idx in (1, 2, 3):
        ge.add_argument(f"--JPA{idx}-F", type=float, required=True)
        ge.add_argument(f"--JPA{idx}-EC", type=float, required=True)
        ge.add_argument(f"--JPA{idx}-EJ", type=float, required=True)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "gain-eta":
        fig = plot_gain_eta_from_physical_params(
            Delta=args.Delta,
            kappa=args.kappa,
            gamma=args.gamma,
            drive=args.drive,
            n_levels=args.n_levels,
            theta=args.theta,
            phi=args.phi,
            npts=args.npts,
            STS_EC=args.STS_EC,
            STS_EL=args.STS_EL,
            JPA1_F=args.JPA1_F, JPA1_EC=args.JPA1_EC, JPA1_EJ=args.JPA1_EJ,
            JPA2_F=args.JPA2_F, JPA2_EC=args.JPA2_EC, JPA2_EJ=args.JPA2_EJ,
            JPA3_F=args.JPA3_F, JPA3_EC=args.JPA3_EC, JPA3_EJ=args.JPA3_EJ,
            figsize=tuple(args.figsize),
        )
        fig.show()
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")