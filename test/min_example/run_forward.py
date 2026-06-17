from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def resolve_executable(mcx: str) -> str:
    mcx_path = Path(mcx)
    if mcx_path.exists():
        return str(mcx_path.resolve())
    mcx_exe = shutil.which(mcx)
    if mcx_exe is None:
        raise SystemExit(f"could not find {mcx!r} on PATH")
    return mcx_exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MCX forward simulation.")
    parser.add_argument("--config", type=Path, default=Path("configs/case.json"))
    parser.add_argument("--session", default="fwd")
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--mcx", default="mcxcl")
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="extra argument passed through to the MCXCL executable; repeat for multiple arguments",
    )
    args = parser.parse_args()

    mcx_exe = resolve_executable(args.mcx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    cmd = [
        mcx_exe,
        "-f",
        str(config_path),
        "-n",
        str(args.photons),
        "-d",
        "1",
        "-w",
        "DSP",
        "-F",
        "jnii",
        "-s",
        args.session,
        *args.mcx_arg,
    ]
    subprocess.run(cmd, cwd=args.output_dir, check=True)


if __name__ == "__main__":
    main()
