#!/usr/bin/env python3
"""Prepare a Digimouse Jacobian experiment scaffold.

This is intentionally conservative: full source-detector Jacobian stacks are
large, so this script writes geometry, masks, fluorescence truth, and MCX
configs without launching the expensive simulations by default.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FACE_DIRS = {
    "+x": (0, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    "-x": (0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    "+y": (1, [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
    "-y": (1, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "+z": (2, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
    "-z": (2, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class OptodeGrid:
    face: str
    positions: list[list[float]]
    directions: list[list[float]]


def load_json_relaxed(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    def compact_zip(match: re.Match[str]) -> str:
        key, data = match.group(1), match.group(2)
        compact = re.sub(r"\s+", "", data)
        return f'"{key}": "{compact}"'

    fixed = re.sub(r'"(_ArrayZipData_)"\s*:\s*"([^"]*)"', compact_zip, text, flags=re.S)
    return json.loads(fixed)


def decode_shapes(config: dict) -> np.ndarray:
    shapes = config["Shapes"]
    if shapes.get("_ArrayZipType_", "zlib").lower() != "zlib":
        raise ValueError("Only zlib-compressed Shapes arrays are supported")
    shape = tuple(int(v) for v in shapes["_ArraySize_"][:3])
    raw = zlib.decompress(base64.b64decode(re.sub(r"\s+", "", shapes["_ArrayZipData_"])))
    expected = math.prod(shape)
    if len(raw) != expected:
        raise ValueError(f"decoded {len(raw)} label bytes, expected {expected}")
    # The atlas JSON uses z-fast flattening: (x * sy + y) * sz + z.
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape, order="C").copy()


def grid_on_face(
    dim: tuple[int, int, int],
    face: str,
    rows: int,
    cols: int,
    margin: int,
) -> OptodeGrid:
    if face not in FACE_DIRS:
        raise ValueError(f"unknown face {face!r}; expected one of {sorted(FACE_DIRS)}")
    normal_axis, face_unit, direction = FACE_DIRS[face]
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    limits = [dim[axis] for axis in plane_axes]
    counts = [cols, rows]
    coords = []
    for limit, count in zip(limits, counts):
        lo = float(margin)
        hi = float(limit - 1 - margin)
        if count == 1:
            coords.append(np.array([(lo + hi) / 2.0], dtype=np.float64))
        else:
            coords.append(np.linspace(lo, hi, count, dtype=np.float64))

    face_coord = float(dim[normal_axis] - 1) if face.startswith("+") else 0.0
    positions: list[list[float]] = []
    directions: list[list[float]] = []
    for b in coords[1]:
        for a in coords[0]:
            pos = [0.0, 0.0, 0.0]
            pos[normal_axis] = face_coord
            pos[plane_axes[0]] = float(a)
            pos[plane_axes[1]] = float(b)
            positions.append(pos)
            directions.append(direction[:])
    return OptodeGrid(face=face, positions=positions, directions=directions)


def gaussian_truth(
    labels: np.ndarray,
    medium_label: int,
    sigma: float,
    amplitude: float,
    seed: int,
    margin: int,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    mask = labels == medium_label
    if margin > 0:
        inner = np.zeros_like(mask)
        inner[margin:-margin, margin:-margin, margin:-margin] = True
        mask &= inner
    candidates = np.argwhere(mask)
    if candidates.size == 0:
        candidates = np.argwhere(labels > 0)
    if candidates.size == 0:
        raise ValueError("could not place Gaussian: label volume is empty")

    center = candidates[int(rng.integers(candidates.shape[0]))]
    axes = np.ogrid[tuple(slice(0, size) for size in labels.shape)]
    dist2 = sum((axes[axis] - float(center[axis])) ** 2 for axis in range(3))
    truth = amplitude * np.exp(-dist2 / (2.0 * sigma * sigma))
    truth *= labels == medium_label
    return truth.astype(np.float32), [int(v) for v in center]


def medium_property(labels: np.ndarray, media: list[dict], name: str) -> np.ndarray:
    values = np.array([float(medium[name]) for medium in media], dtype=np.float32)
    if int(labels.max()) >= len(values):
        raise ValueError("label volume references media not present in Domain.Media")
    return values[labels]


def build_forward_config(
    base: dict,
    source_pos: list[float],
    source_dir: list[float],
    detectors: OptodeGrid,
    session: str,
    photons: int,
    detector_radius: float,
) -> dict:
    config = copy.deepcopy(base)
    config["Session"]["ID"] = session
    config["Session"]["Photons"] = int(photons)
    config["Session"]["DoSaveSeed"] = True
    config["Session"]["DoPartialPath"] = True
    config["Session"]["OutputFormat"] = "jnii"
    config["Session"]["OutputType"] = "F"
    config["Optode"]["Source"] = {
        "Type": "pencil",
        "Pos": source_pos,
        "Dir": [*source_dir, 0.0],
    }
    config["Optode"]["Detector"] = [
        {"Pos": pos, "R": detector_radius}
        for pos in detectors.positions
    ]
    return config


def build_adjoint_config(
    base: dict,
    detector_pos: list[float],
    session: str,
    photons: int,
) -> dict:
    config = copy.deepcopy(base)
    config["Session"]["ID"] = session
    config["Session"]["Photons"] = int(photons)
    config["Session"]["DoSaveSeed"] = False
    config["Session"]["DoPartialPath"] = False
    config["Session"]["OutputFormat"] = "jnii"
    config["Session"]["OutputType"] = "F"
    config["Optode"]["Source"] = {
        "Type": "isotropic",
        "Pos": detector_pos,
        "Dir": [0.0, 0.0, 1.0, 0.0],
    }
    config["Optode"]["Detector"] = []
    return config


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_configs(
    base: dict,
    sources: OptodeGrid,
    detectors: OptodeGrid,
    config_dir: Path,
    photons: int,
    detector_radius: float,
) -> None:
    for src_idx, (src_pos, src_dir) in enumerate(zip(sources.positions, sources.directions), start=1):
        write_json(
            config_dir / f"source_{src_idx:03d}.json",
            build_forward_config(base, src_pos, src_dir, detectors, f"src_{src_idx:03d}", photons, detector_radius),
        )
    for det_idx, det_pos in enumerate(detectors.positions, start=1):
        write_json(
            config_dir / f"adjoint_det_{det_idx:03d}.json",
            build_adjoint_config(base, det_pos, f"adj_det_{det_idx:03d}", photons),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("digimouse.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("jacobian_outputs"))
    parser.add_argument("--config-dir", type=Path, default=Path("jacobian_configs"))
    parser.add_argument("--source-face", choices=sorted(FACE_DIRS), default="+z")
    parser.add_argument("--detector-face", choices=sorted(FACE_DIRS), default="-z")
    parser.add_argument("--source-grid", nargs=2, type=int, metavar=("ROWS", "COLS"), default=(3, 3))
    parser.add_argument("--detector-grid", nargs=2, type=int, metavar=("ROWS", "COLS"), default=(3, 3))
    parser.add_argument("--grid-margin", type=int, default=16)
    parser.add_argument("--detector-radius", type=float, default=4.0)
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--medium-label", type=int, default=1)
    parser.add_argument("--truth-sigma", type=float, default=8.0)
    parser.add_argument("--truth-amplitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = load_json_relaxed(args.config)
    labels = decode_shapes(base)
    dim = tuple(int(v) for v in base["Domain"]["Dim"][:3])
    if labels.shape != dim:
        raise ValueError(f"label shape {labels.shape} does not match Domain.Dim {dim}")

    sources = grid_on_face(dim, args.source_face, args.source_grid[0], args.source_grid[1], args.grid_margin)
    detectors = grid_on_face(dim, args.detector_face, args.detector_grid[0], args.detector_grid[1], args.grid_margin)
    truth, truth_center = gaussian_truth(
        labels,
        args.medium_label,
        args.truth_sigma,
        args.truth_amplitude,
        args.seed,
        args.grid_margin,
    )
    medium_mask = labels == args.medium_label
    mua = medium_property(labels, base["Domain"]["Media"], "mua")
    mus = medium_property(labels, base["Domain"]["Media"], "mus")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "medium_mask.npy", medium_mask)
    np.save(args.output_dir / "mua_by_voxel.npy", mua)
    np.save(args.output_dir / "mus_by_voxel.npy", mus)
    np.save(args.output_dir / "fluorescence_truth.npy", truth)
    write_configs(base, sources, detectors, args.config_dir, args.photons, args.detector_radius)

    metadata = {
        "description": "Digimouse source-detector Jacobian experiment scaffold",
        "shape": list(dim),
        "length_unit_mm": float(base["Domain"].get("LengthUnit", 1.0)),
        "medium_label": args.medium_label,
        "source_face": sources.face,
        "detector_face": detectors.face,
        "source_positions": sources.positions,
        "source_directions": sources.directions,
        "detector_positions": detectors.positions,
        "detector_radius": args.detector_radius,
        "truth": {
            "kind": "Gaussian fluorescence concentration",
            "center_index": truth_center,
            "sigma_voxels": args.truth_sigma,
            "amplitude": args.truth_amplitude,
            "seed": args.seed,
        },
        "jacobian_conventions": {
            "absorption_replay": "MCX replay OutputType J per detector/source, masked to medium_label for coefficient perturbations",
            "scattering_replay": "MCX replay P/mus - J per detector/source, masked to medium_label",
            "fluorescence_replay": "pmcxcl/MCX fluorescence replay with muaf/muf volumes; compare to G_exc * G_em_adj",
            "absorption_twofield": "G_source * G_detector_adjoint",
            "scattering_twofield": "dot(grad(G_source), grad(G_detector_adjoint)) / (3 * (1-g) * mus^2)",
            "fluorescence_twofield": "G_excitation_source * G_emission_detector_adjoint",
        },
        "files": {
            "labels": str(args.output_dir / "labels.npy"),
            "medium_mask": str(args.output_dir / "medium_mask.npy"),
            "fluorescence_truth": str(args.output_dir / "fluorescence_truth.npy"),
            "configs": str(args.config_dir),
        },
    }
    write_json(args.output_dir / "experiment.json", metadata)

    print(f"shape          {dim}")
    print(f"sources        {len(sources.positions)} on {sources.face}")
    print(f"detectors      {len(detectors.positions)} on {detectors.face}")
    print(f"medium label   {args.medium_label} ({int(np.count_nonzero(medium_mask))} voxels)")
    print(f"truth center   {truth_center}")
    print(f"saved          {args.output_dir / 'experiment.json'}")
    print(f"saved configs  {args.config_dir}")


if __name__ == "__main__":
    main()
