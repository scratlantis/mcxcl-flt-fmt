from __future__ import annotations

import argparse
import json
from pathlib import Path


DETECTORS = [
    {"Pos": [15.0, 15.0, 39.0], "R": 1.5},
    {"Pos": [20.0, 15.0, 39.0], "R": 1.5},
    {"Pos": [25.0, 15.0, 39.0], "R": 1.5},
    {"Pos": [15.0, 20.0, 39.0], "R": 1.5},
    {"Pos": [20.0, 20.0, 39.0], "R": 1.5},
    {"Pos": [25.0, 20.0, 39.0], "R": 1.5},
    {"Pos": [15.0, 25.0, 39.0], "R": 1.5},
    {"Pos": [20.0, 25.0, 39.0], "R": 1.5},
    {"Pos": [25.0, 25.0, 39.0], "R": 1.5},
]


def build_config(
    mua_inclusion: float,
    mus_inclusion: float,
    session: str,
    photons: int,
    seed: int,
    dim: tuple[int, int, int] = (40, 40, 40),
) -> dict:
    return {
        "Session": {
            "ID": session,
            "Photons": photons,
            "RNGSeed": seed,
            "DoNormalize": True,
            "DoSaveSeed": True,
            "DoSaveExit": False,
            "OutputFormat": "jnii",
            "OutputType": "F",
        },
        "Forward": {"T0": 0.0, "T1": 5.0e-9, "Dt": 5.0e-9},
        "Domain": {
            "MediaFormat": "byte",
            "LengthUnit": 1.0,
            "Dim": list(dim),
            "OriginType": 1,
            "Media": [
                {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},
                {"mua": 0.01, "mus": 10.0, "g": 0.9, "n": 1.37},
                {"mua": mua_inclusion, "mus": mus_inclusion, "g": 0.9, "n": 1.37},
            ],
        },
        "Optode": {
            "Source": {
                "Type": "pencil",
                "Pos": [20.0, 20.0, 0.0],
                "Dir": [0.0, 0.0, 1.0, 0.0],
            },
            "Detector": DETECTORS,
        },
        "Shapes": [
            {"Grid": {"Tag": 1, "Size": list(dim)}},
            {"Sphere": {"Tag": 2, "O": [20.0, 20.0, 20.0], "R": 5.0}},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a minimal MCX JSON config.")
    parser.add_argument("--mua-inclusion", type=float, default=0.01)
    parser.add_argument("--mus-inclusion", type=float, default=10.0)
    parser.add_argument("--session", default="fwd")
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=29012392)
    parser.add_argument("--output", type=Path, default=Path("configs/case.json"))
    args = parser.parse_args()

    config = build_config(args.mua_inclusion, args.mus_inclusion, args.session, args.photons, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
