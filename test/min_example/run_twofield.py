from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from mcx_io import (
    load_jnii_array,
    measurements_from_jdat,
    measurements_from_mch,
    mock_measurements,
)
from optimize import load_mua_inclusion, load_replay_jacobians


def field3d(path: Path) -> np.ndarray:
    field = np.squeeze(load_jnii_array(path))
    if field.ndim != 3:
        raise ValueError(f"expected a 3-D field in {path}, got shape {field.shape}")
    return field.astype(np.float64)


def comparison_mask(shape: tuple[int, int, int], z_min: int = 0) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if z_min > 0:
        mask[:, :, :z_min] = False
    return mask


def finite_mask(*arrays: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    mask = (
        np.ones(arrays[0].shape, dtype=bool)
        if base is None
        else np.broadcast_to(base, arrays[0].shape).copy()
    )
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def normalized(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return np.zeros(values.shape, dtype=np.float64)
    scale = float(np.nanmax(np.abs(values[valid])))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros(values.shape, dtype=np.float64)
    scaled = values.astype(np.float64) / scale
    return np.where(valid if mask is not None else np.isfinite(scaled), scaled, np.nan)


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = finite_mask(a, b, base=mask)
    if not np.any(valid):
        return float("nan")
    a_norm = normalized(a, valid)
    b_norm = normalized(b, valid)
    a_vals = a_norm[valid]
    b_vals = b_norm[valid]
    denom = float(np.sqrt(np.sum(a_vals * a_vals) * np.sum(b_vals * b_vals)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a_vals * b_vals) / denom)


def least_squares_scale(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = finite_mask(source, target, base=mask)
    if not np.any(valid):
        return 1.0
    source_vals = source[valid]
    target_vals = target[valid]
    denom = float(np.dot(source_vals, source_vals))
    if denom == 0.0 or not np.isfinite(denom):
        return 1.0
    return float(np.dot(source_vals, target_vals) / denom)


def gradient_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gradients_a = np.gradient(a)
    gradients_b = np.gradient(b)
    return sum(ga * gb for ga, gb in zip(gradients_a, gradients_b))


def mus_adjoint_factor(volume_path: Path, config: dict) -> np.ndarray:
    labels = np.load(volume_path)
    factor_by_label = []
    for medium in config["Domain"]["Media"]:
        mus = float(medium["mus"])
        one_minus_g = 1.0 - float(medium["g"])
        if mus > 0.0 and one_minus_g > 0.0:
            factor_by_label.append(1.0 / (3.0 * one_minus_g * mus * mus))
        else:
            factor_by_label.append(0.0)
    factors = np.array(factor_by_label, dtype=np.float64)
    if int(labels.max()) >= len(factors):
        raise ValueError(f"{volume_path} contains labels not defined in Domain.Media")
    return factors[labels.astype(np.int64)]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_source_config(base_config: dict, session: str) -> dict:
    config = copy.deepcopy(base_config)
    config["Session"]["ID"] = session
    config["Session"]["OutputFormat"] = "jnii"
    config["Session"]["OutputType"] = "F"
    return config


def build_adjoint_config(base_config: dict, det_idx: int, session: str) -> dict:
    detector = base_config["Optode"]["Detector"][det_idx]
    config = copy.deepcopy(base_config)
    config["Session"]["ID"] = session
    config["Session"]["OutputFormat"] = "jnii"
    config["Session"]["OutputType"] = "F"
    config["Optode"]["Source"] = {
        "Type": "isotropic",
        "Pos": detector["Pos"],
        "Dir": [0.0, 0.0, 1.0],
    }
    config["Optode"].pop("Detector", None)
    return config


def run_mcx(
    mcx: str,
    config_path: Path,
    photons: int,
    output_dir: Path,
    session: str,
    extra_args: list[str],
) -> None:
    mcx_path = Path(mcx)
    mcx_exe = str(mcx_path.resolve()) if mcx_path.exists() else shutil.which(mcx)
    if mcx_exe is None:
        raise FileNotFoundError(mcx)
    cmd = [
        mcx_exe,
        "-f",
        str(config_path.resolve()),
        "-n",
        str(photons),
        "-d",
        "1",
        "-w",
        "DSP",
        "-F",
        "jnii",
        "-s",
        session,
        *extra_args,
    ]
    subprocess.run(cmd, cwd=output_dir, check=True)


def split_total_photons(total_photons: int, field_count: int) -> int:
    photons = total_photons // field_count
    if photons <= 0:
        raise ValueError(f"--total-photons must be at least {field_count}")
    if photons * field_count != total_photons:
        raise ValueError(f"--total-photons must be divisible by {field_count}")
    return photons


def load_residuals(args: argparse.Namespace, det_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mua = load_mua_inclusion(args.config)
    obs_mua = load_mua_inclusion(args.obs_config)
    if args.mock:
        y_pred = mock_measurements(mua)
        y_obs = mock_measurements(obs_mua)
        return y_pred, y_obs, y_pred - y_obs

    pred_mua_by_label = np.array([0.0, 0.01, mua], dtype=np.float64)
    obs_mua_by_label = np.array([0.0, 0.01, obs_mua], dtype=np.float64)
    pred_reader = measurements_from_mch if args.pred_detp.suffix == ".mch" else measurements_from_jdat
    obs_reader = measurements_from_mch if args.obs_detp.suffix == ".mch" else measurements_from_jdat
    y_pred = pred_reader(args.pred_detp, pred_mua_by_label, det_count)
    y_obs = obs_reader(args.obs_detp, obs_mua_by_label, det_count)
    return y_pred, y_obs, y_pred - y_obs


def property_from_volume(volume_path: Path, config: dict, name: str) -> np.ndarray:
    labels = np.load(volume_path)
    media_values = np.array([medium[name] for medium in config["Domain"]["Media"]], dtype=np.float64)
    if int(labels.max()) >= len(media_values):
        raise ValueError(f"{volume_path} contains labels not defined in Domain.Media")
    return media_values[labels.astype(np.int64)]


def absorption_from_volume(volume_path: Path, config: dict) -> np.ndarray:
    return property_from_volume(volume_path, config, "mua")


def scattering_from_volume(volume_path: Path, config: dict) -> np.ndarray:
    return property_from_volume(volume_path, config, "mus")


def replay_mus_components(
    output_dir: Path,
    volume_path: Path,
    config: dict,
    det_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pathlength = load_replay_jacobians(output_dir, det_count, "J")
    scat_counts = load_replay_jacobians(output_dir, det_count, "P")
    mus = scattering_from_volume(volume_path, config)
    scat_over_mus = np.divide(scat_counts, mus, out=np.zeros_like(scat_counts), where=mus[None, ...] > 0.0)
    return pathlength, scat_over_mus, scat_over_mus - pathlength


def replay_mus_jacobians(output_dir: Path, volume_path: Path, config: dict, det_count: int) -> np.ndarray:
    _, _, jacobians = replay_mus_components(output_dir, volume_path, config, det_count)
    return jacobians


def save_field_visualization(
    source: np.ndarray,
    detector_fields: list[np.ndarray],
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).parent / ".matplotlib").resolve()))
    import matplotlib.pyplot as plt

    z = source.shape[2] // 2
    fig, axes = plt.subplots(1, 1 + len(detector_fields), figsize=(3.2 * (1 + len(detector_fields)), 3.2))
    axes = np.atleast_1d(axes)

    images = [source] + detector_fields
    titles = ["G source"] + [f"G det {idx}" for idx in range(1, len(detector_fields) + 1)]
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(np.log10(np.maximum(image[:, :, z].T, 1e-30)), origin="lower", cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_gradient_comparison(
    replay_gradient: np.ndarray,
    twofield_gradient: np.ndarray,
    output: Path,
    truth: np.ndarray | None = None,
    truth_title: str = "truth",
    mask: np.ndarray | None = None,
    robust_percentile: float = 100.0,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).parent / ".matplotlib").resolve()))
    import matplotlib.pyplot as plt

    z = replay_gradient.shape[2] // 2
    replay_norm = normalized(replay_gradient, mask)
    twofield_norm = normalized(twofield_gradient, mask)
    diff = twofield_norm - replay_norm

    cols = 4 if truth is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(3.35 * cols, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    panels = [
        (replay_norm[:, :, z], "replay grad", "coolwarm"),
        (twofield_norm[:, :, z], "two-field grad", "coolwarm"),
        (diff[:, :, z], "two-field - replay", "coolwarm"),
    ]
    if truth is not None:
        panels.insert(0, (truth[:, :, z], truth_title, "viridis"))

    for axis, (image, title, cmap) in zip(axes, panels):
        finite = np.isfinite(image)
        if title != truth_title and np.any(finite) and robust_percentile < 100.0:
            vmax = max(float(np.nanpercentile(np.abs(image[finite]), robust_percentile)), 1e-12)
        else:
            vmax = max(float(np.nanmax(np.abs(image))), 1e-12)
        if title == truth_title:
            shown = axis.imshow(image.T, origin="lower", cmap=cmap)
        else:
            shown = axis.imshow(image.T, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_mus_jacobian_diagnostics(
    replay_pathlength: np.ndarray,
    replay_scat_over_mus: np.ndarray,
    replay_jacobians: np.ndarray,
    twofield_jacobians: np.ndarray,
    calibrated_twofield_jacobians: np.ndarray,
    output: Path,
    mask: np.ndarray,
    robust_percentile: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).parent / ".matplotlib").resolve()))
    import matplotlib.pyplot as plt

    z = replay_jacobians.shape[3] // 2
    det_count = replay_jacobians.shape[0]
    fig, axes = plt.subplots(det_count, 5, figsize=(15.5, 3.1 * det_count), constrained_layout=True)
    axes = np.atleast_2d(axes)

    columns = [
        ("replay J", replay_pathlength, "viridis", False),
        ("replay P/mus", replay_scat_over_mus, "viridis", False),
        ("replay mus J", replay_jacobians, "coolwarm", True),
        ("two-field raw", twofield_jacobians, "coolwarm", True),
        ("two-field scaled", calibrated_twofield_jacobians, "coolwarm", True),
    ]
    for det_idx in range(det_count):
        for col_idx, (title, stack, cmap, signed) in enumerate(columns):
            image = stack[det_idx, :, :, z].astype(np.float64)
            valid = np.isfinite(stack[det_idx]) & mask
            shown_image = np.where(np.isfinite(image), image, np.nan)
            axis = axes[det_idx, col_idx]
            if signed:
                vals = stack[det_idx][valid]
                vmax = 1e-12
                if vals.size:
                    vmax = max(float(np.nanpercentile(np.abs(vals), robust_percentile)), vmax)
                shown = axis.imshow(shown_image.T, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                vals = stack[det_idx][valid]
                vmax = 1e-12
                if vals.size:
                    vmax = max(float(np.nanpercentile(vals, robust_percentile)), vmax)
                shown = axis.imshow(shown_image.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
            axis.set_title(f"det {det_idx + 1} {title}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_mus_diagnostic_report(
    output: Path,
    replay_pathlength: np.ndarray,
    replay_scat_over_mus: np.ndarray,
    replay_jacobians: np.ndarray,
    raw_twofield_jacobians: np.ndarray,
    calibrated_twofield_jacobians: np.ndarray,
    mask: np.ndarray,
    twofield_scale: float,
) -> None:
    report = {
        "mask": {
            "valid_voxels_per_detector": int(np.count_nonzero(mask)),
            "total_voxels_per_detector": int(mask.size),
        },
        "twofield_scale": twofield_scale,
        "detectors": [],
    }
    for det_idx in range(replay_jacobians.shape[0]):
        valid = finite_mask(replay_jacobians[det_idx], calibrated_twofield_jacobians[det_idx], base=mask)
        bad_replay = (~np.isfinite(replay_jacobians[det_idx])) | (np.abs(replay_jacobians[det_idx]) > 1e3)
        entry = {
            "detector": det_idx + 1,
            "valid_comparison_voxels": int(np.count_nonzero(valid)),
            "bad_replay_voxels": int(np.count_nonzero(bad_replay)),
            "bad_replay_z_min": None,
            "bad_replay_z_max": None,
        }
        bad_locations = np.argwhere(bad_replay)
        if bad_locations.size:
            entry["bad_replay_z_min"] = int(bad_locations[:, 2].min())
            entry["bad_replay_z_max"] = int(bad_locations[:, 2].max())

        for name, array in [
            ("replay_pathlength", replay_pathlength[det_idx]),
            ("replay_scat_over_mus", replay_scat_over_mus[det_idx]),
            ("replay_mus_jacobian", replay_jacobians[det_idx]),
            ("twofield_raw_mus_jacobian", raw_twofield_jacobians[det_idx]),
            ("twofield_scaled_mus_jacobian", calibrated_twofield_jacobians[det_idx]),
        ]:
            vals = array[valid]
            entry[name] = {
                "min": float(np.nanmin(vals)) if vals.size else None,
                "max": float(np.nanmax(vals)) if vals.size else None,
                "mean": float(np.nanmean(vals)) if vals.size else None,
                "abs_p99": float(np.nanpercentile(np.abs(vals), 99.0)) if vals.size else None,
                "abs_max": float(np.nanmax(np.abs(vals))) if vals.size else None,
            }

        entry["corr_scaled"] = correlation(
            replay_jacobians[det_idx],
            calibrated_twofield_jacobians[det_idx],
            valid,
        )
        report["detectors"].append(entry)

    output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a two-field adjoint MCX gradient pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/case.json"))
    parser.add_argument("--obs-config", type=Path, default=Path("configs/obs.json"))
    parser.add_argument("--pred-detp", type=Path, default=Path("outputs/fwd_detp.jdat"))
    parser.add_argument("--obs-detp", type=Path, default=Path("outputs/obs_detp.jdat"))
    parser.add_argument("--volume", type=Path, default=Path("data/volume.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--photons", type=int, default=1_000_000, help="photons per source/adjoint MCX run")
    parser.add_argument(
        "--total-photons",
        type=int,
        help="overall two-field photon budget, split equally across source plus detector fields",
    )
    parser.add_argument("--mcx", default="mcxcl")
    parser.add_argument(
        "--mcx-arg",
        action="append",
        default=[],
        help="extra argument passed through to the MCXCL executable; repeat for multiple arguments",
    )
    parser.add_argument("--property", choices=("mua", "mus"), default="mua")
    parser.add_argument(
        "--mask-z-min",
        type=int,
        help="exclude voxels below this z index when comparing/calibrating; defaults to 3 for mus, 0 for mua",
    )
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=99.5,
        help="percentile used for comparison plot color limits",
    )
    parser.add_argument(
        "--no-calibrate-twofield",
        action="store_true",
        help="disable scalar least-squares calibration of two-field mus Jacobians to replay",
    )
    parser.add_argument(
        "--twofield-sign",
        type=float,
        default=1.0,
        help="sign applied to the two-field Jacobian; default matches the current mua replay convention",
    )
    parser.add_argument("--skip-mcx", action="store_true", help="reuse existing G_source/G_det_i jnii files")
    parser.add_argument("--mock", action="store_true", help="use deterministic mock measurements for residuals")
    args = parser.parse_args()

    if not args.skip_mcx and shutil.which(args.mcx) is None:
        mcx_path = Path(args.mcx)
        if not mcx_path.exists():
            raise SystemExit(f"could not find {args.mcx!r} on PATH")

    base_config = json.loads(args.config.read_text())
    detectors = base_config["Optode"].get("Detector", [])
    if not detectors:
        raise ValueError(f"{args.config} does not define Optode.Detector entries")
    photons_per_field = args.photons
    if args.total_photons is not None:
        photons_per_field = split_total_photons(args.total_photons, 1 + len(detectors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_config_path = args.config_dir / "source.json"
    source_session = "G_source"
    write_json(source_config_path, build_source_config(base_config, source_session))

    adjoint_config_paths = []
    for det_idx in range(len(detectors)):
        session = f"G_det_{det_idx + 1}"
        path = args.config_dir / f"adj_det_{det_idx + 1}.json"
        write_json(path, build_adjoint_config(base_config, det_idx, session))
        adjoint_config_paths.append(path)

    if not args.skip_mcx:
        run_mcx(args.mcx, source_config_path, photons_per_field, args.output_dir, source_session, args.mcx_arg)
        for det_idx, config_path in enumerate(adjoint_config_paths, start=1):
            run_mcx(args.mcx, config_path, photons_per_field, args.output_dir, f"G_det_{det_idx}", args.mcx_arg)

    source = field3d(args.output_dir / "G_source.jnii")
    detector_fields = [
        field3d(args.output_dir / f"G_det_{det_idx}.jnii")
        for det_idx in range(1, len(detectors) + 1)
    ]
    base_config_for_property = json.loads(args.config.read_text())
    if args.property == "mua":
        jacobians = np.stack([args.twofield_sign * source * detector for detector in detector_fields], axis=0)
    else:
        factor = mus_adjoint_factor(args.volume, base_config_for_property)
        jacobians = np.stack(
            [args.twofield_sign * gradient_dot(source, detector) * factor for detector in detector_fields],
            axis=0,
        )

    mask_z_min = args.mask_z_min
    if mask_z_min is None:
        mask_z_min = 3 if args.property == "mus" else 0
    mask = comparison_mask(jacobians.shape[1:], mask_z_min)

    y_pred, y_obs, residual = load_residuals(args, len(detectors))
    twofield_scale = 1.0
    replay_pathlength = None
    replay_scat_over_mus = None
    if args.property == "mua":
        replay_jacobians = load_replay_jacobians(args.output_dir, len(detectors), "J")
        truth = absorption_from_volume(args.volume, json.loads(args.obs_config.read_text()))
        truth_title = "truth mua"
        output_stem = "twofield"
    else:
        replay_pathlength, replay_scat_over_mus, replay_jacobians = replay_mus_components(
            args.output_dir,
            args.volume,
            base_config_for_property,
            len(detectors),
        )
        if not args.no_calibrate_twofield:
            twofield_scale = least_squares_scale(jacobians, replay_jacobians, mask[None, ...])
            jacobians = jacobians * twofield_scale
        truth = scattering_from_volume(args.volume, json.loads(args.obs_config.read_text()))
        truth_title = "truth mus"
        output_stem = "twofield_mus"

    comparison_valid = finite_mask(replay_jacobians, jacobians, base=mask[None, ...])
    twofield_gradient = np.tensordot(residual, jacobians, axes=(0, 0))
    replay_gradient = np.tensordot(residual, replay_jacobians, axes=(0, 0))
    gradient_mask = finite_mask(replay_gradient, twofield_gradient, base=mask)

    np.save(args.output_dir / f"{output_stem}_jacobians.npy", jacobians)
    np.save(args.output_dir / f"replay_{args.property}_jacobians.npy", replay_jacobians)
    np.save(args.output_dir / f"{output_stem}_gradient.npy", twofield_gradient)
    np.save(args.output_dir / f"replay_{args.property}_gradient_from_residual.npy", replay_gradient)
    np.save(args.output_dir / f"{output_stem}_difference_normalized.npy", normalized(twofield_gradient, gradient_mask) - normalized(replay_gradient, gradient_mask))
    np.save(args.output_dir / f"{output_stem}_comparison_mask.npy", mask)

    corr = correlation(replay_gradient, twofield_gradient, gradient_mask)
    try:
        save_field_visualization(source, detector_fields, args.output_dir / "twofield_fields.png")
        save_gradient_comparison(
            replay_gradient,
            twofield_gradient,
            args.output_dir / f"{output_stem}_comparison.png",
            truth,
            truth_title,
            gradient_mask,
            args.robust_percentile,
        )
        if args.property == "mus" and replay_pathlength is not None and replay_scat_over_mus is not None:
            raw_twofield_jacobians = jacobians / twofield_scale if twofield_scale != 0.0 else jacobians
            save_mus_jacobian_diagnostics(
                replay_pathlength,
                replay_scat_over_mus,
                replay_jacobians,
                raw_twofield_jacobians,
                jacobians,
                args.output_dir / "mus_jacobian_diagnostics.png",
                mask,
                args.robust_percentile,
            )
            write_mus_diagnostic_report(
                args.output_dir / "mus_jacobian_diagnostics.json",
                replay_pathlength,
                replay_scat_over_mus,
                replay_jacobians,
                raw_twofield_jacobians,
                jacobians,
                mask,
                twofield_scale,
            )
        saved_plots = True
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        saved_plots = False

    print(f"y_pred       {y_pred}")
    print(f"y_obs        {y_obs}")
    print(f"residual     {residual}")
    print(f"property     {args.property}")
    print(f"corr         {corr:.8f}")
    print(f"mask z>=     {mask_z_min}")
    if args.property == "mus":
        print(f"twofield x   {twofield_scale:.8e}")
        print(f"valid voxels {int(np.count_nonzero(comparison_valid))}/{comparison_valid.size}")
    print(f"budget       {photons_per_field} photons x {1 + len(detectors)} fields")
    print(f"saved        {args.output_dir / f'{output_stem}_gradient.npy'}")
    if saved_plots:
        print(f"saved        {args.output_dir / f'{output_stem}_comparison.png'}")
        print(f"saved        {args.output_dir / 'twofield_fields.png'}")
        if args.property == "mus":
            print(f"saved        {args.output_dir / 'mus_jacobian_diagnostics.png'}")
            print(f"saved        {args.output_dir / 'mus_jacobian_diagnostics.json'}")
    else:
        print("plots        skipped; install matplotlib or use the project venv")


if __name__ == "__main__":
    main()
