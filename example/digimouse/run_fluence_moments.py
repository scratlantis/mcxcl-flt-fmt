#!/usr/bin/env python3
"""Run matched Digimouse fluence and squared-contribution simulations."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def load_domain_shape(path: Path) -> tuple[int, int, int]:
    text = path.read_text()
    try:
        payload = json.loads(text)
        return tuple(int(value) for value in payload["Domain"]["Dim"][:3])
    except json.JSONDecodeError:
        match = re.search(r'"Dim"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', text)
        if not match:
            raise ValueError(f"{path}: could not read Domain.Dim")
        return tuple(int(value) for value in match.groups())


def load_mc2(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: found {values.size} values, expected {expected}")
    # MCX stores x as the fastest-changing spatial coordinate.
    return values.reshape(shape, order="F")


def run_mcx(
    mcx: Path,
    config: Path,
    output_dir: Path,
    session: str,
    output_type: str,
    photons: int,
    seed: int,
    extra_args: list[str],
) -> Path:
    command = [
        str(mcx),
        "-A",
        "-n",
        str(photons),
        "-f",
        str(config),
        "-F",
        "mc2",
        "-D",
        "P",
        "-E",
        str(seed),
        "-U",
        "0",
        "-O",
        output_type,
        "-s",
        session,
        "--root",
        str(output_dir),
        *extra_args,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=output_dir, check=True)
    result = output_dir / f"{session}.mc2"
    if not result.exists():
        raise FileNotFoundError(f"MCX completed but did not create {result}")
    return result


def derive_volumes(
    fluence_path: Path,
    moment2_path: Path,
    shape: tuple[int, int, int],
    output_dir: Path,
    relative_threshold: float,
) -> dict[str, float | int | list[int]]:
    fluence = load_mc2(fluence_path, shape)
    moment2 = load_mc2(moment2_path, shape)
    if fluence.shape != moment2.shape:
        raise ValueError(f"shape mismatch: {fluence.shape} != {moment2.shape}")

    max_fluence = float(np.nanmax(fluence))
    threshold = max_fluence * relative_threshold
    valid = np.isfinite(fluence) & np.isfinite(moment2) & (fluence > threshold) & (moment2 >= 0.0)

    fluence64 = fluence.astype(np.float64)
    moment64 = moment2.astype(np.float64)
    fluence_sq = fluence64 * fluence64
    ratio = np.zeros(fluence.shape, dtype=np.float32)
    effective = np.zeros(fluence.shape, dtype=np.float32)
    np.divide(moment64, fluence_sq, out=ratio, where=valid)
    np.divide(fluence_sq, moment64, out=effective, where=valid & (moment2 > 0.0))

    np.save(output_dir / "fluence.npy", fluence)
    np.save(output_dir / "fluence_moment2.npy", moment2)
    np.save(output_dir / "fluence_contribution_ratio.npy", ratio)
    np.save(output_dir / "fluence_effective_contributions.npy", effective)

    ratio_valid = ratio[valid]
    above_one = int(np.count_nonzero(ratio_valid > 1.0 + 1e-4))
    return {
        "shape": list(fluence.shape),
        "relative_threshold": relative_threshold,
        "absolute_threshold": threshold,
        "valid_voxels": int(np.count_nonzero(valid)),
        "ratio_min": float(np.min(ratio_valid)) if ratio_valid.size else 0.0,
        "ratio_max": float(np.max(ratio_valid)) if ratio_valid.size else 0.0,
        "ratio_voxels_above_one": above_one,
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcx", type=Path, default=(here / "../../bin/mcxcl").resolve())
    parser.add_argument("--config", type=Path, default=here / "digimouse.json")
    parser.add_argument("--output-dir", type=Path, default=here / "fluence_moments")
    parser.add_argument("--photons", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--relative-threshold", type=float, default=1e-12)
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="additional argument passed to both MCX runs; repeat as needed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    mcx = args.mcx.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = load_domain_shape(config)

    fluence_path = output_dir / "digimouse_fluence.mc2"
    moment2_path = output_dir / "digimouse_fluence_moment2.mc2"
    if not args.process_only:
        fluence_path = run_mcx(
            mcx, config, output_dir, "digimouse_fluence", "F", args.photons, args.seed, args.mcx_arg
        )
        moment2_path = run_mcx(
            mcx, config, output_dir, "digimouse_fluence_moment2", "K", args.photons, args.seed, args.mcx_arg
        )

    stats = derive_volumes(fluence_path, moment2_path, shape, output_dir, args.relative_threshold)
    metadata = {
        "config": str(config),
        "mcx": str(mcx),
        "photons": args.photons,
        "seed": args.seed,
        "fluence_mc2": str(fluence_path),
        "moment2_mc2": str(moment2_path),
        **stats,
    }
    (output_dir / "fluence_moments.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {output_dir / 'fluence_contribution_ratio.npy'}")
    print(
        f"valid voxels {stats['valid_voxels']}, ratio range "
        f"[{stats['ratio_min']:.6g}, {stats['ratio_max']:.6g}]"
    )
    if stats["ratio_voxels_above_one"]:
        print(
            f"warning: {stats['ratio_voxels_above_one']} voxels exceed 1 by more than 1e-4; "
            "inspect low-fluence noise and atomic summation"
        )


if __name__ == "__main__":
    main()
