from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


OUTPUT_TYPES = {
    "mua": [("J", "J")],
    "mus": [("J", "J"), ("P", "P")],
    "both": [("J", "J"), ("P", "P")],
}


def resolve_executable(mcx: str) -> str:
    mcx_path = Path(mcx)
    if mcx_path.exists():
        return str(mcx_path.resolve())
    mcx_exe = shutil.which(mcx)
    if mcx_exe is None:
        raise SystemExit(f"could not find {mcx!r} on PATH")
    return mcx_exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run replay outputs per detector.")
    parser.add_argument("--config", type=Path, default=Path("configs/case.json"))
    parser.add_argument("--seed", type=Path, default=Path("outputs/fwd_detp.jdat"))
    parser.add_argument("--detectors", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--mcx", default="mcxcl")
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="extra argument passed through to the MCXCL executable; repeat for multiple arguments",
    )
    parser.add_argument(
        "--property",
        choices=sorted(OUTPUT_TYPES),
        default="mua",
        help="mua writes J_det_*.jnii; mus/both also write P_det_*.jnii for scattering counts",
    )
    args = parser.parse_args()

    mcx_exe = resolve_executable(args.mcx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    seed_path = args.seed.resolve()
    for output_type, prefix in OUTPUT_TYPES[args.property]:
        for det_idx in range(1, args.detectors + 1):
            cmd = [
                mcx_exe,
                "-f",
                str(config_path),
                "-E",
                str(seed_path),
                "-O",
                output_type,
                "-Y",
                str(det_idx),
                "-F",
                "jnii",
                "-s",
                f"{prefix}_det_{det_idx}",
                *args.mcx_arg,
            ]
            subprocess.run(cmd, cwd=args.output_dir, check=True)


if __name__ == "__main__":
    main()
