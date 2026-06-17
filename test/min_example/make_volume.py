from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DIM = (40, 40, 40)
DEFAULT_CENTER = (20, 20, 20)
DEFAULT_RADIUS = 5.0


def make_volume(
    dim: tuple[int, int, int] = DEFAULT_DIM,
    center: tuple[float, float, float] = DEFAULT_CENTER,
    radius: float = DEFAULT_RADIUS,
) -> np.ndarray:
    """Return a labeled cube: 1 background tissue, 2 spherical inclusion."""
    volume = np.ones(dim, dtype=np.uint8)
    grid = np.indices(dim, dtype=np.float32)
    dist2 = sum((grid[axis] - center[axis]) ** 2 for axis in range(3))
    volume[dist2 <= radius**2] = 2
    return volume


def parse_triplet(value: str, cast: type = int) -> tuple:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return tuple(cast(part) for part in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the minimal labeled MCX volume.")
    parser.add_argument("--dim", type=lambda s: parse_triplet(s, int), default=DEFAULT_DIM)
    parser.add_argument(
        "--center", type=lambda s: parse_triplet(s, float), default=DEFAULT_CENTER
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--output", type=Path, default=Path("data/volume.npy"))
    args = parser.parse_args()

    volume = make_volume(args.dim, args.center, args.radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, volume)
    print(f"saved {args.output} with shape {volume.shape} and labels {np.unique(volume)}")


if __name__ == "__main__":
    main()
