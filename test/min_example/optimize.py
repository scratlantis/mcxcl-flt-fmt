from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mcx_io import (
    load_jnii_array,
    measurements_from_jdat,
    measurements_from_mch,
    mock_jacobians,
    mock_measurements,
)


def load_mua_inclusion(config_path: Path) -> float:
    config = json.loads(config_path.read_text())
    return float(config["Domain"]["Media"][2]["mua"])


def load_mus_inclusion(config_path: Path) -> float:
    config = json.loads(config_path.read_text())
    return float(config["Domain"]["Media"][2]["mus"])


def load_property_inclusion(config_path: Path, name: str) -> float:
    config = json.loads(config_path.read_text())
    return float(config["Domain"]["Media"][2][name])


def load_replay_jacobians(output_dir: Path, det_count: int, prefix: str = "J") -> np.ndarray:
    arrays = []
    for det_idx in range(1, det_count + 1):
        path = output_dir / f"{prefix}_det_{det_idx}.jnii"
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run run_replay.py with --property "
                f"{'mus' if prefix == 'P' else 'mua'} first"
            )
        arrays.append(load_jnii_array(path).squeeze())
    return np.stack(arrays, axis=0)


def property_from_volume(volume: np.ndarray, config_path: Path, name: str) -> np.ndarray:
    config = json.loads(config_path.read_text())
    media_values = np.array([medium[name] for medium in config["Domain"]["Media"]], dtype=np.float64)
    if int(volume.max()) >= len(media_values):
        raise ValueError(f"{config_path} does not define all labels in the volume")
    return media_values[volume.astype(np.int64)]


def load_replay_mus_jacobians(output_dir: Path, volume: np.ndarray, config_path: Path, det_count: int) -> np.ndarray:
    pathlength = load_replay_jacobians(output_dir, det_count, "J")
    scat_counts = load_replay_jacobians(output_dir, det_count, "P")
    mus = property_from_volume(volume, config_path, "mus")
    return np.divide(scat_counts, mus, out=np.zeros_like(scat_counts), where=mus[None, ...] > 0.0) - pathlength


def one_step(
    y_pred: np.ndarray,
    y_obs: np.ndarray,
    jacobians: np.ndarray,
    inclusion_mask: np.ndarray,
    mua: float,
    lr: float,
) -> dict:
    residual = y_pred - y_obs
    loss = float(np.sum(residual**2))
    grad_map = np.tensordot(residual, jacobians, axes=(0, 0))
    scalar_grad = float(grad_map[inclusion_mask].mean())
    updated_mua = max(0.0, mua - lr * scalar_grad)
    return {
        "y_pred": y_pred,
        "y_obs": y_obs,
        "residual": residual,
        "loss": loss,
        "grad_map": grad_map,
        "scalar_grad": scalar_grad,
        "updated_mua": updated_mua,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Accumulate replay gradients and take one optical-property step.")
    parser.add_argument("--config", type=Path, default=Path("configs/case.json"))
    parser.add_argument("--obs-config", type=Path, default=Path("configs/obs.json"))
    parser.add_argument("--pred-detp", type=Path, default=Path("outputs/fwd_detp.jdat"))
    parser.add_argument("--obs-detp", type=Path, default=Path("outputs/obs_detp.jdat"))
    parser.add_argument("--volume", type=Path, default=Path("data/volume.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--detectors", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--property", choices=("mua", "mus"), default="mua")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    volume = np.load(args.volume)
    inclusion_mask = volume == 2
    current_value = load_property_inclusion(args.config, args.property)

    if args.mock:
        if args.property != "mua":
            raise SystemExit("--mock currently only supports --property mua")
        obs_mua = load_mua_inclusion(args.obs_config)
        y_pred = mock_measurements(current_value)
        y_obs = mock_measurements(obs_mua)
        jacobians = mock_jacobians(volume)
    else:
        obs_mua = load_mua_inclusion(args.obs_config)
        pred_mua = load_mua_inclusion(args.config)
        pred_mua_by_label = np.array([0.0, 0.01, pred_mua], dtype=np.float64)
        obs_mua_by_label = np.array([0.0, 0.01, obs_mua], dtype=np.float64)
        pred_reader = measurements_from_mch if args.pred_detp.suffix == ".mch" else measurements_from_jdat
        obs_reader = measurements_from_mch if args.obs_detp.suffix == ".mch" else measurements_from_jdat
        y_pred = pred_reader(args.pred_detp, pred_mua_by_label, args.detectors)
        y_obs = obs_reader(args.obs_detp, obs_mua_by_label, args.detectors)
        if args.property == "mua":
            jacobians = load_replay_jacobians(args.output_dir, args.detectors, "J")
        else:
            jacobians = load_replay_mus_jacobians(args.output_dir, volume, args.config, args.detectors)

    result = one_step(y_pred, y_obs, jacobians, inclusion_mask, current_value, args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "gradient.npy" if args.property == "mua" else f"gradient_{args.property}.npy"
    np.save(args.output_dir / output_name, result["grad_map"])

    print(f"y_pred       {result['y_pred']}")
    print(f"y_obs        {result['y_obs']}")
    print(f"residual     {result['residual']}")
    print(f"loss         {result['loss']:.8e}")
    print(f"scalar_grad  {result['scalar_grad']:.8e}")
    print(f"property     {args.property}")
    print(f"{args.property} update   {current_value:.8f} -> {result['updated_mua']:.8f}")
    print(f"saved        {args.output_dir / output_name}")


if __name__ == "__main__":
    main()
