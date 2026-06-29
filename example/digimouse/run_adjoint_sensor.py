#!/usr/bin/env python3
"""Run adjoint Monte Carlo for a detector grid; save per-detector masked fluence.

Workflow:
  1. Load an existing forward fluence volume and a boolean marked-voxel mask.
  2. Place a grid of detectors on the specified face using isotropic point sources
     (adjoint principle: detector → source).
  3. For each detector, run MCX, extract the fluence only at marked voxels, then
     delete the full volume to keep disk usage low.
  4. Save masked_forward.npy [n_marked] and masked_adjoint.npy [n_dets, n_marked].

The sensor measurement for detector d is the dot product
  M_d = sum_v( phi_fwd[v] * phi_adj_d[v] )
which view_sensor.py computes from the saved arrays.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np


FACE_DIRS: dict[str, tuple[int, list[float]]] = {
    "+x": (0, [-1.0, 0.0, 0.0]),
    "-x": (0, [1.0, 0.0, 0.0]),
    "+y": (1, [0.0, -1.0, 0.0]),
    "-y": (1, [0.0, 1.0, 0.0]),
    "+z": (2, [0.0, 0.0, -1.0]),
    "-z": (2, [0.0, 0.0, 1.0]),
}

# Short aliases so the shell doesn't mistake -z etc. for flags.
FACE_ALIASES: dict[str, str] = {
    "px": "+x", "mx": "-x",
    "py": "+y", "my": "-y",
    "pz": "+z", "mz": "-z",
}


def parse_face(value: str) -> str:
    canonical = FACE_ALIASES.get(value, value)
    if canonical not in FACE_DIRS:
        choices = list(FACE_DIRS) + list(FACE_ALIASES)
        raise argparse.ArgumentTypeError(
            f"invalid face {value!r}; expected one of {sorted(choices)}"
        )
    return canonical


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


def decode_shapes(config: dict) -> np.ndarray:
    """Decode the atlas label volume from the base config's Shapes field."""
    shapes = config["Shapes"]
    shape = tuple(int(v) for v in shapes["_ArraySize_"][:3])
    raw = zlib.decompress(base64.b64decode(re.sub(r"\s+", "", shapes["_ArrayZipData_"])))
    if len(raw) != math.prod(shape):
        raise ValueError(f"decoded {len(raw)} label bytes, expected {math.prod(shape)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape, order="C").copy()


def decode_jdata_array(node: dict) -> np.ndarray:
    type_map = {
        "single": "<f4", "double": "<f8",
        "uint32": "<u4", "int32": "<i4",
        "uint16": "<u2", "int16": "<i2",
        "uint8": "u1",  "int8": "i1",
    }
    raw = base64.b64decode(node["_ArrayZipData_"])
    if node.get("_ArrayZipType_") == "zlib":
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=np.dtype(type_map[node["_ArrayType_"]])).reshape(
        tuple(node["_ArraySize_"])
    )


def load_detp_ppaths(path: Path) -> np.ndarray | None:
    """Read ppath from a MCX _detp.jdat; returns float64 [n_photons, n_tissue_media] or None."""
    try:
        photon_data = load_json_relaxed(path)["MCXData"]["PhotonData"]
    except Exception as exc:
        print(f"  jdat parse error: {exc}")
        return None
    if "ppath" not in photon_data:
        return None
    ppaths = decode_jdata_array(photon_data["ppath"]).astype(np.float64)
    if ppaths.ndim == 1:
        ppaths = ppaths.reshape(1, -1)
    return ppaths if ppaths.ndim == 2 and ppaths.shape[0] > 0 else None


def load_detp_mch_ppaths(path: Path) -> np.ndarray | None:
    """Read ppath from a MCX _detp.mch; returns float64 [n_photons, n_all_media] or None.

    Column 0 = medium 0 (vacuum), columns 1..N = tissue media.
    MCX writes .mch instead of .jdat when OutputFormat is mc2.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"MCXH":
                return None
            f.read(4)                                       # version
            n_media   = struct.unpack("<I", f.read(4))[0]  # maxmedia
            f.read(4)                                       # detnum
            col_count = struct.unpack("<I", f.read(4))[0]  # colcount
            f.read(4)                                       # totalphoton
            f.read(4)                                       # detected
            n_saved   = struct.unpack("<I", f.read(4))[0]  # savedphoton
            f.read(32)                                      # rest of 64-byte header
            raw = np.frombuffer(f.read(), dtype=np.float32)
        if n_saved == 0 or raw.size == 0:
            return None
        data = raw.reshape(n_saved, col_count)
        return data[:, 1 : 1 + n_media].astype(np.float64)  # skip detid col 0
    except Exception as exc:
        print(f"  mch read error: {exc}")
        return None


def load_mc2(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"{path}: found {values.size} values, expected {expected}")
    return values.reshape(shape, order="F")


def grid_on_face(
    dim: tuple[int, int, int],
    face: str,
    rows: int,
    cols: int,
    margin: int,
    spread: float = 1.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Build a rows×cols optode grid on a face.

    spread=1.0 spans the full face (minus margin); spread=0.0 collapses all
    detectors to the face centre; intermediate values scale linearly.
    """
    if face not in FACE_DIRS:
        raise ValueError(f"unknown face {face!r}; expected one of {sorted(FACE_DIRS)}")
    normal_axis, direction = FACE_DIRS[face]
    plane_axes = [a for a in range(3) if a != normal_axis]
    coords = []
    for axis, count in zip(plane_axes, [cols, rows]):
        lo, hi = float(margin), float(dim[axis] - 1 - margin)
        centre = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * spread
        lo_s, hi_s = centre - half, centre + half
        coords.append(np.array([centre]) if count == 1 else np.linspace(lo_s, hi_s, count))
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
    return positions, directions


def build_adjoint_config(
    base: dict,
    det_pos: list[float],
    session_id: str,
    photons: int,
    seed: int,
    sphere_detector: dict,
) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("Session", {}).update(
        {
            "ID": session_id,
            "Photons": photons,
            "RNGSeed": seed,
            "DoSaveSeed": False,
            "DoPartialPath": True,
            "DoSaveVolume": True,
            "DoNormalize": True,
            "OutputFormat": "mc2",
            "OutputType": "F",
        }
    )
    config.setdefault("Forward", {}).update({"T0": 0.0, "T1": 5e-9, "Dt": 5e-9})
    config["Optode"]["Source"] = {
        "Type": "isotropic",
        "Pos": det_pos,
        "Dir": [0.0, 0.0, 1.0, 0.0],
    }
    config["Optode"]["Detector"] = [sphere_detector]
    return config


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcx", type=Path, default=(here / "../../bin/mcxcl").resolve())
    parser.add_argument("--config", type=Path, default=here / "digimouse.json")
    parser.add_argument(
        "--fluence",
        type=Path,
        default=here / "fluence_moments/fluence.npy",
        help="forward fluence volume (3D float32 .npy)",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="boolean marked-voxel mask (3D bool .npy, same shape as fluence)",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "adjoint_sensor_output")
    parser.add_argument(
        "--detector-face",
        type=parse_face,
        default="-z",
        metavar="{+x,+y,+z,-x,-y,-z,px,py,pz,mx,my,mz}",
        help="face on which to place the detector grid (use px/mx/... to avoid shell flag confusion)",
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--margin", type=int, default=16, help="voxel margin from face edges")
    parser.add_argument(
        "--spread",
        type=float,
        default=1.0,
        metavar="[0,1]",
        help="detector grid spread: 1=full face, 0=all detectors at face centre",
    )
    parser.add_argument("--photons", type=int, default=100_000, help="photons per adjoint run")
    parser.add_argument("--seed", type=int, default=20260629, help="base RNG seed (incremented per detector)")
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="extra argument forwarded to MCX; repeat as needed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcx = args.mcx.resolve()
    config_path = args.config.resolve()
    fluence_path = args.fluence.resolve()
    mask_path = args.mask.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_json_relaxed(config_path)
    shape = tuple(int(v) for v in base["Domain"]["Dim"][:3])

    mask = np.load(mask_path).astype(bool)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match Domain.Dim {shape}")
    n_marked = int(np.count_nonzero(mask))
    if n_marked == 0:
        raise ValueError("mask has no marked voxels")
    print(f"marked voxels  {n_marked:,}")

    fluence = np.load(fluence_path)
    if fluence.shape != shape:
        raise ValueError(f"fluence shape {fluence.shape} does not match Domain.Dim {shape}")
    masked_forward = fluence[mask].astype(np.float32)
    del fluence

    media = base["Domain"]["Media"]
    n_media = len(media)
    mua_orig = np.array([float(m["mua"]) for m in media])  # mm⁻¹

    # fwd_mean weights adjoint photon paths by forward fluence at the ROI.
    fwd_mean = float(masked_forward.mean()) if masked_forward.size > 0 else 1.0

    # Sphere detector centred on ROI — adjoint photons passing through it get
    # their partial paths recorded (written to _detp.jdat or _detp.mch by MCX).
    marked_coords = np.argwhere(mask).astype(np.float64)
    centroid = marked_coords.mean(axis=0)
    bounding_r = float(np.max(np.linalg.norm(marked_coords - centroid, axis=1)))
    sphere_r = max(bounding_r + 10.0, 10.0)
    sphere_detector = {"Pos": centroid.tolist(), "R": sphere_r}
    print(f"sphere: centroid={[f'{v:.1f}' for v in centroid]}  "
          f"bounding_r={bounding_r:.1f}  detector_r={sphere_r:.1f}  fwd_mean={fwd_mean:.4g}")

    spread = max(0.0, min(1.0, args.spread))
    positions, _ = grid_on_face(shape, args.detector_face, args.rows, args.cols, args.margin, spread)
    n_dets = args.rows * args.cols
    det_positions_grid = np.array(positions, dtype=np.float32).reshape(args.rows, args.cols, 3)

    masked_adjoint = np.zeros((n_dets, n_marked), dtype=np.float32)
    path_sens_adj = np.zeros((n_dets, n_media), dtype=np.float64)

    for det_idx, det_pos in enumerate(positions):
        row = det_idx // args.cols
        col = det_idx % args.cols
        session_id = f"adj_det_{det_idx:04d}"
        seed = args.seed + det_idx

        config = build_adjoint_config(base, det_pos, session_id, args.photons, seed, sphere_detector)
        config_file = output_dir / f"{session_id}.json"
        config_file.write_text(json.dumps(config, indent=2) + "\n")

        command = [
            str(mcx),
            "-A",
            "-n", str(args.photons),
            "-f", str(config_file),
            "-F", "mc2",
            "-D", "P",
            "-E", str(seed),
            "-U", "1",
            "-O", "F",
            "-s", session_id,
            "--root", str(output_dir),
            *args.mcx_arg,
        ]
        print(f"\n[{det_idx + 1}/{n_dets}] detector ({row}, {col}) at {[f'{v:.1f}' for v in det_pos]}", flush=True)
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=output_dir, check=True)

        mc2_path = output_dir / f"{session_id}.mc2"
        adj_fluence = load_mc2(mc2_path, shape)
        masked_adjoint[det_idx] = adj_fluence[mask].astype(np.float32)
        mc2_path.unlink()

        # Path-length sensitivity from adjoint photons detected by the sphere.
        # path_sens[d, m] = fwd_mean * sum_photons( w * ppath_m )
        # where w = exp(-sum_m(mua_m * ppath_m)) is the Beer-Lambert survival weight.
        jdat_path = output_dir / f"{session_id}_detp.jdat"
        mch_path  = next(
            (output_dir / name for name in (f"{session_id}_detp.mch", f"{session_id}.mch")
             if (output_dir / name).exists()),
            output_dir / f"{session_id}_detp.mch",  # default (non-existent) for the check below
        )
        # Show what MCX actually wrote so we can diagnose missing files
        written = [p.name for p in sorted(output_dir.glob(f"{session_id}*")) if p != config_file]
        print(f"  written: {written}")

        ppaths = col_offset = None
        if jdat_path.exists():
            ppaths = load_detp_ppaths(jdat_path)
            col_offset = 1          # jdat excludes medium 0 (vacuum)
            jdat_path.unlink()
        elif mch_path.exists():
            ppaths = load_detp_mch_ppaths(mch_path)
            col_offset = 0          # mch includes medium 0
            mch_path.unlink()

        if ppaths is not None and ppaths.shape[0] > 0:
            n_cols = ppaths.shape[1]
            mua_slice = mua_orig[col_offset : col_offset + n_cols]
            weights = np.exp(-(ppaths * mua_slice).sum(axis=1))
            path_sens_adj[det_idx, col_offset : col_offset + n_cols] = (
                fwd_mean * (weights[:, None] * ppaths).sum(axis=0)
            )
            print(f"  {ppaths.shape[0]} photons  sens_sum={path_sens_adj[det_idx].sum():.4g}")

    masked_forward_path = output_dir / "masked_forward.npy"
    masked_adjoint_path = output_dir / "masked_adjoint.npy"
    det_positions_path = output_dir / "detector_positions.npy"
    mask_copy_path = output_dir / "marked_mask.npy"
    path_sens_adj_path = output_dir / "path_sens_adj.npy"

    np.save(masked_forward_path, masked_forward)
    np.save(masked_adjoint_path, masked_adjoint)
    np.save(det_positions_path, det_positions_grid)
    np.save(mask_copy_path, mask)

    nz = int(np.count_nonzero(path_sens_adj))
    print(f"sensitivity: nonzero {nz}/{path_sens_adj.size}  "
          f"range [{path_sens_adj.min():.4g}, {path_sens_adj.max():.4g}]")
    np.save(path_sens_adj_path, path_sens_adj)

    metadata = {
        "base_config": str(config_path),
        "fluence": str(fluence_path),
        "mask": str(mask_path),
        "shape": list(shape),
        "n_marked_voxels": n_marked,
        "detector_face": args.detector_face,
        "rows": args.rows,
        "cols": args.cols,
        "margin": args.margin,
        "spread": spread,
        "photons_per_detector": args.photons,
        "base_seed": args.seed,
        "n_detectors": n_dets,
        "mua_orig": mua_orig.tolist(),
        "masked_forward": str(masked_forward_path),
        "masked_adjoint": str(masked_adjoint_path),
        "detector_positions": str(det_positions_path),
        "marked_mask": str(mask_copy_path),
        "path_sens_adj": str(path_sens_adj_path),
        "sens_method": "born_global",
    }
    (output_dir / "adjoint_sensor_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\ndone: {n_dets} detectors, {n_marked:,} marked voxels")
    print(f"saved {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nadjoint sensor simulation cancelled", file=sys.stderr)
        raise SystemExit(130)
