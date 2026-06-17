from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def mcx_arg_options(values: list[str]) -> list[str]:
    return [f"--mcx-arg={value}" for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run replay and two-field comparison with equal gradient photon budgets."
    )
    parser.add_argument("--photons", type=int, required=True, help="overall photon budget per gradient method")
    parser.add_argument("--mcx", default="mcxcl")
    parser.add_argument("--property", choices=("mua", "mus"), default="mua")
    parser.add_argument("--obs-mua-inclusion", type=float, default=0.03)
    parser.add_argument("--fwd-mua-inclusion", type=float, default=0.01)
    parser.add_argument("--obs-mus-inclusion", type=float, default=10.0)
    parser.add_argument("--fwd-mus-inclusion", type=float, default=10.0)
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="extra argument passed through to the MCXCL executable; repeat for multiple arguments",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python = sys.executable
    mcx_path = Path(args.mcx)
    if not mcx_path.is_absolute() and (root / mcx_path).exists():
        mcx = str((root / mcx_path).resolve())
    else:
        mcx = args.mcx

    run([python, "make_volume.py"], root)
    run(
        [
            python,
            "make_config.py",
            "--mua-inclusion",
            str(args.obs_mua_inclusion),
            "--mus-inclusion",
            str(args.obs_mus_inclusion),
            "--session",
            "obs",
            "--photons",
            str(args.photons),
            "--output",
            "configs/obs.json",
        ],
        root,
    )
    run(
        [
            python,
            "make_config.py",
            "--mua-inclusion",
            str(args.fwd_mua_inclusion),
            "--mus-inclusion",
            str(args.fwd_mus_inclusion),
            "--session",
            "fwd",
            "--photons",
            str(args.photons),
            "--output",
            "configs/case.json",
        ],
        root,
    )
    run(
        [
            python,
            "run_forward.py",
            "--config",
            "configs/obs.json",
            "--session",
            "obs",
            "--photons",
            str(args.photons),
            "--mcx",
            mcx,
            *mcx_arg_options(args.mcx_arg),
        ],
        root,
    )
    run(
        [
            python,
            "run_forward.py",
            "--config",
            "configs/case.json",
            "--session",
            "fwd",
            "--photons",
            str(args.photons),
            "--mcx",
            mcx,
            *mcx_arg_options(args.mcx_arg),
        ],
        root,
    )
    run(
        [
            python,
            "run_replay.py",
            "--config",
            "configs/case.json",
            "--seed",
            "outputs/fwd_detp.jdat",
            "--detectors",
            "9",
            "--mcx",
            mcx,
            "--property",
            "both" if args.property == "mus" else "mua",
            *mcx_arg_options(args.mcx_arg),
        ],
        root,
    )
    run(
        [
            python,
            "run_twofield.py",
            "--config",
            "configs/case.json",
            "--obs-config",
            "configs/obs.json",
            "--total-photons",
            str(args.photons),
            "--mcx",
            mcx,
            "--property",
            args.property,
            *mcx_arg_options(args.mcx_arg),
        ],
        root,
    )


if __name__ == "__main__":
    main()
