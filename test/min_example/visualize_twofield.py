from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_twofield import (
    absorption_from_volume,
    field3d,
    save_field_visualization,
    save_gradient_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved two-field MCX outputs.")
    parser.add_argument("--obs-config", type=Path, default=Path("configs/obs.json"))
    parser.add_argument("--volume", type=Path, default=Path("data/volume.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--detectors", type=int, default=9)
    parser.add_argument("--property", choices=("mua", "mus"), default="mua")
    args = parser.parse_args()

    source = field3d(args.output_dir / "G_source.jnii")
    detector_fields = [
        field3d(args.output_dir / f"G_det_{det_idx}.jnii")
        for det_idx in range(1, args.detectors + 1)
    ]
    stem = "twofield" if args.property == "mua" else "twofield_mus"
    replay_gradient = np.load(args.output_dir / f"replay_{args.property}_gradient_from_residual.npy")
    twofield_gradient = np.load(args.output_dir / f"{stem}_gradient.npy")
    truth = absorption_from_volume(args.volume, json.loads(args.obs_config.read_text()))

    try:
        save_field_visualization(source, detector_fields, args.output_dir / "twofield_fields.png")
        save_gradient_comparison(
            replay_gradient,
            twofield_gradient,
            args.output_dir / f"{stem}_comparison.png",
            truth,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit("matplotlib is required for two-field visualization") from exc
        raise

    print(f"saved {args.output_dir / 'twofield_fields.png'}")
    print(f"saved {args.output_dir / f'{stem}_comparison.png'}")


if __name__ == "__main__":
    main()
