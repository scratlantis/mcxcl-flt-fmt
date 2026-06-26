#!/usr/bin/env python3
"""Run a secondary Digimouse flux pass from fluence-weighted marked voxels."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_json_relaxed(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    def compact_zip(match: re.Match[str]) -> str:
        compact = re.sub(r"\s+", "", match.group(2))
        return f'"{match.group(1)}": "{compact}"'

    fixed = re.sub(r'"(_ArrayZipData_)"\s*:\s*"([^"]*)"', compact_zip, text, flags=re.S)
    return json.loads(fixed)


def load_mc2(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"{path}: found {values.size} values, expected {expected}")
    return values.reshape(shape, order="F")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcx", type=Path, default=(here / "../../bin/mcxcl").resolve())
    parser.add_argument("--config", type=Path, default=here / "digimouse.json")
    parser.add_argument(
        "--source-volume",
        type=Path,
        default=here / "fluence_moments/marked_source/source_weights.npy",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "marked_source_output")
    parser.add_argument("--photons", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="additional argument passed to MCX; repeat as needed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcx = args.mcx.resolve()
    config_path = args.config.resolve()
    source_path = args.source_volume.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = np.load(source_path)
    if source.ndim != 3:
        raise ValueError(f"expected a 3D source volume, got {source.shape}")
    if not np.all(np.isfinite(source)) or np.any(source < 0.0):
        raise ValueError("source weights must be finite and nonnegative")
    total_strength = float(np.sum(source, dtype=np.float64))
    if not math.isfinite(total_strength) or total_strength <= 0.0:
        raise ValueError("source weights must have a finite positive sum")

    config = load_json_relaxed(config_path)
    shape = tuple(int(value) for value in config["Domain"]["Dim"][:3])
    if source.shape != shape:
        raise ValueError(f"source shape {source.shape} does not match Domain.Dim {shape}")

    binary_path = output_dir / "source_weights.bin"
    np.asarray(source, dtype="<f4").ravel(order="F").tofile(binary_path)

    session = config.setdefault("Session", {})
    session.update(
        {
            "ID": "digimouse_marked_source_unit",
            "Photons": args.photons,
            "RNGSeed": args.seed,
            "DoSaveVolume": True,
            "DoNormalize": True,
            "OutputFormat": "mc2",
            "OutputType": "F",
        }
    )
    config.setdefault("Forward", {}).update({"T0": 0.0, "T1": 5e-9, "Dt": 5e-9})
    config.setdefault("Optode", {})["Source"] = {
        "Type": "volumetric",
        "Pos": [0.0, 0.0, 0.0, 1.0],
        "Dir": [0.0, 0.0, 1.0, "_NaN_"],
        "Param1": [shape[0], shape[1], shape[2], 0.0],
        "Param2": [0.0, 0.0, 0.0, 0.0],
        "Pattern": {
            "Nx": shape[0],
            "Ny": shape[1],
            "Nz": shape[2],
            "Data": str(binary_path),
        },
    }
    config["Optode"]["Detector"] = []

    generated_config = output_dir / "marked_source.json"
    generated_config.write_text(json.dumps(config, indent=2) + "\n")

    command = [
        str(mcx),
        "-A",
        "-n",
        str(args.photons),
        "-f",
        str(generated_config),
        "-F",
        "mc2",
        "-D",
        "P",
        "-E",
        str(args.seed),
        "-U",
        "1",
        "-O",
        "F",
        "-s",
        "digimouse_marked_source_unit",
        "--root",
        str(output_dir),
        *args.mcx_arg,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=output_dir, check=True)

    mc2_path = output_dir / "digimouse_marked_source_unit.mc2"
    unit_fluence = load_mc2(mc2_path, shape)
    scaled_fluence = (unit_fluence.astype(np.float64) * total_strength).astype(np.float32)
    unit_path = output_dir / "secondary_unit_fluence.npy"
    scaled_path = output_dir / "secondary_scaled_fluence.npy"
    np.save(unit_path, unit_fluence)
    np.save(scaled_path, scaled_fluence)

    metadata_path = source_path.parent / "marked_source.json"
    selection = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    metadata = {
        "base_config": str(config_path),
        "generated_config": str(generated_config),
        "source_volume": str(source_path),
        "source_binary": str(binary_path),
        "shape": list(shape),
        "photons": args.photons,
        "seed": args.seed,
        "total_relative_source_strength": total_strength,
        "scaling": "secondary_scaled_fluence = secondary_unit_fluence * sum(source_weights)",
        "unit_fluence": str(unit_path),
        "scaled_fluence": str(scaled_path),
        "selection": selection,
    }
    (output_dir / "marked_source_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"total relative source strength: {total_strength:.9g}")
    print(f"wrote {scaled_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmarked-source simulation cancelled", file=sys.stderr)
        raise SystemExit(130)
