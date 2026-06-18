from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def import_local_pmcxcl(repo_root: Path):
    sys.path.insert(0, str(repo_root / "pmcxcl"))

    try:
        import pmcxcl  # type: ignore
    except Exception as exc:  # pragma: no cover - this is a task-facing diagnostic
        raise SystemExit(
            "Could not import pmcxcl. Run the VS Code task "
            "'MCX fluo replay: build pmcxcl' first."
        ) from exc

    if not hasattr(pmcxcl, "run"):
        raise SystemExit(
            "pmcxcl imported, but _pmcxcl.run is unavailable. Run the VS Code task "
            "'MCX fluo replay: build pmcxcl' first."
        )

    return pmcxcl


def make_arrays(dim: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = dim
    vol = np.ones(dim, dtype=np.uint8, order="F")

    x, y, z = np.ogrid[:nx, :ny, :nz]
    inclusion = (x - nx * 0.50) ** 2 + (y - ny * 0.50) ** 2 + (z - nz * 0.56) ** 2 <= 5.0**2
    vol[inclusion] = 2

    muaf = np.full(dim, 0.006, dtype=np.float32, order="F")
    muaf[inclusion] = 0.018

    muf = np.zeros(dim, dtype=np.float32, order="F")
    muf[inclusion] = 1.0

    return np.asfortranarray(vol), np.asfortranarray(muaf), np.asfortranarray(muf)


def base_config(dim: tuple[int, int, int], photons: int, seed: int) -> dict[str, Any]:
    nx, ny, nz = dim
    return {
        "nphoton": photons,
        "seed": seed,
        "vol": None,
        "prop": np.asfortranarray(
            np.array(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.01, 3.0, 0.9, 1.37],
                    [0.02, 3.0, 0.9, 1.37],
                ],
                dtype=np.float32,
            )
        ),
        "tstart": 0.0,
        "tend": 5.0e-9,
        "tstep": 5.0e-9,
        "srcpos": [nx * 0.5, ny * 0.5, 0.0],
        "srcdir": [0.0, 0.0, 1.0],
        "detpos": np.asfortranarray(
            np.array([[nx * 0.5, ny * 0.5, nz - 1.0, 8.0]], dtype=np.float32)
        ),
        "issave2pt": True,
        "issavedet": 1,
        "issaveseed": True,
        "savedetflag": "DP",
        "isatomic": 0,
        "isreflect": 1,
        "isspecular": False,
        "isnormalized": 1,
        "maxdetphoton": max(photons, 1024),
        "outputtype": "flux",
        "gpuid": 1,
    }


def run_with_enough_photons(pmcxcl, cfg: dict[str, Any], min_detected: int) -> dict[str, Any]:
    result = pmcxcl.run(**cfg)
    detected = int(result.get("seeds", np.empty((0, 0))).shape[1])

    if detected < min_detected:
        raise SystemExit(
            f"Only {detected} photons reached the detector; need at least {min_detected}. "
            "Increase --photons or detector radius in the script."
        )

    return result


def replay_config(
    cfg: dict[str, Any],
    forward: dict[str, Any],
    muaf: np.ndarray,
    muf: np.ndarray,
    outputtype: str,
) -> dict[str, Any]:
    replay = dict(cfg)
    replay.update(
        {
            "nphoton": int(forward["seeds"].shape[1]),
            "seed": np.asfortranarray(forward["seeds"]),
            "detphotons": np.asfortranarray(forward["detp"]),
            "muaf": muaf,
            "muf": muf,
            "outputtype": outputtype,
            "issavedet": 0,
            "issaveseed": False,
            "issave2pt": True,
        }
    )
    return replay


def collapse_to_volume(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)

    while data.ndim > 3:
        data = data.sum(axis=-1)

    return np.squeeze(data)


def middle_slice(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)

    if data.ndim == 1:
        return data[np.newaxis, :]

    if data.ndim == 2:
        return data

    z = data.shape[2] // 2
    return data[:, :, z]


def draw_heatmap(ax, title: str, arr: np.ndarray, cmap: str = "viridis") -> None:
    data = middle_slice(arr)
    im = ax.imshow(np.asarray(data).T, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def visualize(
    output_png: Path,
    vol: np.ndarray,
    muaf: np.ndarray,
    muf: np.ndarray,
    forward: dict[str, Any],
    jacobian: dict[str, Any],
    fluo: dict[str, Any],
) -> None:
    fwd_flux = collapse_to_volume(forward["flux"])
    jac_flux = collapse_to_volume(jacobian["flux"])
    fluo_flux = collapse_to_volume(fluo["flux"])
    diff = fluo_flux - jac_flux

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    panels = [
        ("volume labels", vol, "tab20"),
        ("muaf input", muaf, "magma"),
        ("muf input", muf, "magma"),
        ("forward flux", fwd_flux, "viridis"),
        ("jacobian replay flux", jac_flux, "viridis"),
        ("fluo replay flux", fluo_flux, "viridis"),
        ("fluo - jacobian", diff, "coolwarm"),
        ("detp records", np.asarray(forward["detp"])[:, :128], "viridis"),
        ("seed bytes", np.asarray(forward["seeds"])[:, :128], "viridis"),
    ]

    for ax, (title, data, cmap) in zip(axes.flat, panels):
        draw_heatmap(ax, title, np.asarray(data), cmap)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def save_npz(
    output_npz: Path,
    vol: np.ndarray,
    muaf: np.ndarray,
    muf: np.ndarray,
    forward: dict[str, Any],
    jacobian: dict[str, Any],
    fluo: dict[str, Any],
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        vol=vol,
        muaf=muaf,
        muf=muf,
        forward_flux=np.asarray(forward["flux"]),
        forward_detp=np.asarray(forward["detp"]),
        forward_seeds=np.asarray(forward["seeds"]),
        jacobian_flux=np.asarray(jacobian["flux"]),
        fluo_flux=np.asarray(fluo["flux"]),
    )


def write_summary(
    output_json: Path,
    forward: dict[str, Any],
    jacobian: dict[str, Any],
    fluo: dict[str, Any],
) -> None:
    def stat_block(result: dict[str, Any]) -> dict[str, float]:
        flux = np.asarray(result["flux"], dtype=np.float64)
        return {
            "flux_sum": float(flux.sum()),
            "flux_min": float(flux.min()),
            "flux_max": float(flux.max()),
            "runtime": float(result["stat"]["runtime"]),
        }

    summary = {
        "detected_photons": int(forward["seeds"].shape[1]),
        "forward": stat_block(forward),
        "jacobian_replay": stat_block(jacobian),
        "fluo_replay": stat_block(fluo),
        "max_abs_fluo_minus_jacobian": float(
            np.max(np.abs(np.asarray(fluo["flux"]) - np.asarray(jacobian["flux"])))
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the fluo replay clone path and visualize buffers.")
    parser.add_argument("--photons", type=int, default=20000)
    parser.add_argument("--min-detected", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29012392)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

    repo_root = Path(__file__).resolve().parents[2]
    pmcxcl = import_local_pmcxcl(repo_root)

    dim = (40, 40, 40)
    vol, muaf, muf = make_arrays(dim)
    cfg = base_config(dim, args.photons, args.seed)
    cfg["vol"] = vol
    cfg["muaf"] = muaf
    cfg["muf"] = muf

    print("running forward pass with detector records and seeds")
    forward = run_with_enough_photons(pmcxcl, cfg, args.min_detected)

    print("running jacobian replay baseline")
    jacobian = pmcxcl.run(**replay_config(cfg, forward, muaf, muf, "jacobian"))

    print("running fluo replay clone path")
    fluo = pmcxcl.run(**replay_config(cfg, forward, muaf, muf, "fluo"))

    output_png = args.output_dir / "fluo_replay_path_buffers.png"
    output_npz = args.output_dir / "fluo_replay_path_buffers.npz"
    output_json = args.output_dir / "fluo_replay_path_summary.json"

    visualize(output_png, vol, muaf, muf, forward, jacobian, fluo)
    save_npz(output_npz, vol, muaf, muf, forward, jacobian, fluo)
    write_summary(output_json, forward, jacobian, fluo)

    print(f"detected photons: {forward['seeds'].shape[1]}")
    print(f"wrote {output_png}")
    print(f"wrote {output_npz}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
