"""
Compare fluorescence Jacobian (∂y_det / ∂muf[k]) computed via:

  Two-field:  J[det, k] = G_exc[k]  ×  G_em_adj[det, k]
  Replay:     J[det, k] = otFluoReplay kernel (new functionality)

Geometry: 40×40×40 mm slab, background tissue (label 1) with a spherical
fluorescent inclusion (label 2, muf > 0) at the centre.  Nine detectors at
z = 39 (same layout as the mua / mus comparison scripts).

Output files in --output-dir:
  fluo_fields.png             – excitation flux and two emission adjoint fluxes
  fluo_comparison.png         – four-panel: truth | replay grad | two-field grad | diff
  fluo_jacobians.npy          – (n_det, NX, NY, NZ) replay Jacobians
  fluo_twofield_jacobians.npy – (n_det, NX, NY, NZ) two-field Jacobians (calibrated)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "pmcxcl"))
import pmcxcl  # type: ignore


# ---------------------------------------------------------------------------
# Geometry (0-indexed, issrcfrom0=1 to match pmcxcl's native convention)
# Matches the 40×40×40 phantom from make_config.py / make_volume.py.
# Detectors are at z = NZ (exit face, 0-indexed).
# ---------------------------------------------------------------------------
NX, NY, NZ = 40, 40, 40

SOURCE_POS = [19.0, 19.0, 0.0]   # 0-indexed centre of 40×40 grid, entrance face
SOURCE_DIR = [0.0, 0.0, 1.0]

DETECTORS = np.array([
    [14.0, 14.0, NZ, 1.5],
    [19.0, 14.0, NZ, 1.5],
    [24.0, 14.0, NZ, 1.5],
    [14.0, 19.0, NZ, 1.5],
    [19.0, 19.0, NZ, 1.5],
    [24.0, 19.0, NZ, 1.5],
    [14.0, 24.0, NZ, 1.5],
    [19.0, 24.0, NZ, 1.5],
    [24.0, 24.0, NZ, 1.5],
], dtype=np.float32, order="F")


# ---------------------------------------------------------------------------
# Volume and optical-property helpers
# ---------------------------------------------------------------------------

def make_volume() -> np.ndarray:
    vol = np.ones((NX, NY, NZ), dtype=np.uint8, order="F")
    grid = np.indices((NX, NY, NZ), dtype=np.float32)
    dist2 = sum((grid[ax] - 20.0) ** 2 for ax in range(3))
    vol[dist2 <= 25.0] = 2  # label-2 sphere of radius 5 centred at index (20,20,20)
    return vol


def make_muf_vol(vol: np.ndarray, muf_inclusion: float) -> np.ndarray:
    muf = np.where(vol == 2, muf_inclusion, 0.0).astype(np.float32)
    return np.asfortranarray(muf)


def make_prop(mua: float, mus: float, g: float, n: float) -> np.ndarray:
    return np.array([
        [0.0, 0.0, 1.0, 1.0],   # label 0: void
        [mua, mus,  g,  n  ],   # label 1: background
        [mua, mus,  g,  n  ],   # label 2: inclusion (same optical props)
    ], dtype=np.float32, order="F")


# ---------------------------------------------------------------------------
# MCX run helpers
# ---------------------------------------------------------------------------

def _base_kwargs(vol, prop, nphoton, gpuid):
    return dict(
        vol=vol, prop=prop, nphoton=nphoton,
        tstart=0.0, tend=5e-9, tstep=5e-9,
        issrcfrom0=1,   # 0-indexed coordinates throughout
        isnormalized=0, isatomic=1, autopilot=1, gpuid=gpuid,
    )


def run_forward(vol, prop, nphoton, gpuid) -> dict:
    return pmcxcl.run(
        **_base_kwargs(vol, prop, nphoton, gpuid),
        srcpos=SOURCE_POS, srcdir=SOURCE_DIR,
        detpos=DETECTORS,
        issavedet=1, issaveseed=1, savedetflag="DP",
        outputtype="flux",
    )


def run_emission_adjoint(vol, prop_em, det_row, nphoton, gpuid) -> np.ndarray:
    """Run from a single detector (isotropic source) with emission properties."""
    result = pmcxcl.run(
        **_base_kwargs(vol, prop_em, nphoton, gpuid),
        srcpos=[float(det_row[0]), float(det_row[1]), float(det_row[2])],
        srctype="isotropic",
        srcdir=[0.0, 0.0, 1.0],
        issavedet=0, issaveseed=0,
        outputtype="flux",
    )
    return np.squeeze(np.asarray(result["flux"]))


def run_fluo_replay(vol, prop, muaf_vol, muf_vol, prop_muaf,
                    seeds, detp, det_idx, gpuid) -> np.ndarray:
    """Replay excitation seeds and compute fluorescence Jacobian for one detector."""
    result = pmcxcl.run(
        **_base_kwargs(vol, prop, seeds.shape[1], gpuid),
        muaf=muaf_vol, muf=muf_vol, prop_muaf=prop_muaf,
        seed=seeds, detphotons=detp,
        srcpos=SOURCE_POS, srcdir=SOURCE_DIR,
        detpos=DETECTORS,
        issavedet=0, issaveseed=0,
        outputtype="fluo",
        replaydet=det_idx,
    )
    return np.squeeze(np.asarray(result["flux"]))


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def normalized(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(values) & (mask if mask is not None else True)
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float64)
    scale = float(np.nanmax(np.abs(values[valid])))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(values, dtype=np.float64)
    out = values.astype(np.float64) / scale
    return np.where(valid, out, np.nan)


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b) & mask
    if not np.any(valid):
        return float("nan")
    an = normalized(a, valid)
    bn = normalized(b, valid)
    av, bv = an[valid], bn[valid]
    denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    return float(np.dot(av, bv) / denom) if denom else float("nan")


def least_squares_scale(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(source) & np.isfinite(target) & mask
    if not np.any(valid):
        return 1.0
    sv, tv = source[valid], target[valid]
    denom = float(np.dot(sv, sv))
    return float(np.dot(sv, tv) / denom) if denom else 1.0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _mpl():
    os.environ.setdefault("MPLCONFIGDIR",
                          str((ROOT / ".matplotlib").resolve()))
    import matplotlib.pyplot as plt
    return plt


def save_field_visualization(
    G_exc: np.ndarray,
    G_em_adjs: list[np.ndarray],
    det_indices: list[int],
    output: Path,
) -> None:
    plt = _mpl()
    z = NZ // 2
    images = [G_exc] + G_em_adjs
    titles = ["G_exc (flux)"] + [f"G_em_adj det {i + 1}" for i in det_indices]
    fig, axes = plt.subplots(1, len(images), figsize=(3.3 * len(images), 3.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.log10(np.maximum(img[:, :, z].T, 1e-30)),
                  origin="lower", cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_gradient_comparison(
    replay_grad: np.ndarray,
    twofield_grad: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    corr: float,
    output: Path,
    robust_percentile: float = 99.5,
) -> None:
    plt = _mpl()
    z = NZ // 2
    rn = normalized(replay_grad, mask)
    tn = normalized(twofield_grad, mask)
    diff = tn - rn
    panels = [
        (truth[:, :, z],  "truth muf",           "viridis", False),
        (rn[:, :, z],     "replay gradient",      "coolwarm", True),
        (tn[:, :, z],     "two-field gradient",   "coolwarm", True),
        (diff[:, :, z],   "two-field − replay",   "coolwarm", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), constrained_layout=True)
    for ax, (img, title, cmap, signed) in zip(axes, panels):
        finite = np.isfinite(img)
        if signed and np.any(finite):
            vmax = max(float(np.nanpercentile(np.abs(img[finite]), robust_percentile)), 1e-12)
            shown = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            shown = ax.imshow(img.T, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Fluorescence Jacobian comparison  (corr = {corr:.4f})", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fluorescence Jacobian via two-field and replay."
    )
    parser.add_argument("--mua",  type=float, default=0.01,
                        help="excitation absorption coefficient mm⁻¹ (default 0.01)")
    parser.add_argument("--muaf", type=float, default=0.02,
                        help="emission absorption coefficient mm⁻¹ (default 0.02)")
    parser.add_argument("--mus",  type=float, default=10.0,
                        help="scattering coefficient mm⁻¹ (default 10.0)")
    parser.add_argument("--g",    type=float, default=0.9,
                        help="anisotropy factor (default 0.9)")
    parser.add_argument("--n",    type=float, default=1.37,
                        help="refractive index (default 1.37)")
    parser.add_argument("--muf-inclusion", type=float, default=0.1,
                        help="fluorophore concentration in inclusion (default 0.1)")
    parser.add_argument("--photons", type=int, default=1_000_000,
                        help="photons per MCX run (default 1 000 000)")
    parser.add_argument("--adjoint-photons", type=int, default=0,
                        help="photons per emission-adjoint run (0 = same as --photons)")
    parser.add_argument("--gpuid",   type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--robust-percentile", type=float, default=99.5)
    args = parser.parse_args()

    adj_photons = args.adjoint_photons or args.photons

    # --- Build volume and optical properties ---
    vol = make_volume()
    inclusion = (vol == 2)

    prop_exc = make_prop(args.mua,  args.mus, args.g, args.n)
    prop_em  = make_prop(args.muaf, args.mus, args.g, args.n)

    muaf_vol  = np.full(vol.shape, args.muaf, dtype=np.float32, order="F")
    muf_vol   = np.zeros(vol.shape, dtype=np.float32, order="F")
    prop_muaf = np.array([0.0, args.muaf, args.muaf], dtype=np.float32)

    n_det = len(DETECTORS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Forward excitation pass ---
    print(f"Running excitation forward pass ({args.photons:,} photons) ...")
    fwd = run_forward(vol, prop_exc, args.photons, args.gpuid)
    seeds  = fwd["seeds"]
    if "detp" not in fwd:
        raise SystemExit("No photons detected — check geometry and detector positions.")
    detp   = fwd["detp"]
    G_exc  = np.squeeze(np.asarray(fwd["flux"])).astype(np.float64)
    N_det_total = seeds.shape[1]
    print(f"  Detected {N_det_total:,} photons across {n_det} detectors.")

    # --- Emission adjoint passes (one per detector) ---
    print(f"Running {n_det} emission adjoint passes ({adj_photons:,} photons each) ...")
    G_em_adjs: list[np.ndarray] = []
    for det_idx, det_row in enumerate(DETECTORS):
        print(f"  Adjoint detector {det_idx + 1}/{n_det} ...", end=" ", flush=True)
        G_adj = run_emission_adjoint(vol, prop_em, det_row, adj_photons, args.gpuid)
        G_em_adjs.append(G_adj.astype(np.float64))
        print("done")

    # --- Fluorescence Jacobian via replay (one run per detector) ---
    print(f"Running {n_det} fluorescence replay passes ...")
    replay_jacobians = np.zeros((n_det, NX, NY, NZ), dtype=np.float64)
    for det_idx in range(n_det):
        print(f"  Replay detector {det_idx + 1}/{n_det} ...", end=" ", flush=True)
        replay_jacobians[det_idx] = run_fluo_replay(
            vol, prop_exc, muaf_vol, muf_vol, prop_muaf,
            seeds, detp, det_idx + 1, args.gpuid,
        )
        print("done")

    # --- Two-field Jacobian: G_exc × G_em_adj ---
    twofield_jacobians_raw = np.stack(
        [G_exc * G_adj for G_adj in G_em_adjs], axis=0
    )  # (n_det, NX, NY, NZ)

    # Calibrate two-field scale to replay via least-squares over the inclusion
    mask = np.ones((NX, NY, NZ), dtype=bool)
    scale = least_squares_scale(
        twofield_jacobians_raw.ravel(),
        replay_jacobians.ravel(),
        np.broadcast_to(mask, (n_det, NX, NY, NZ)).ravel(),
    )
    twofield_jacobians = twofield_jacobians_raw * scale

    # --- Gradient: weight Jacobians by fluorescence signal from inclusion ---
    # residual[det] = muf_inclusion * sum(J_replay[det, inclusion])
    residual = np.array([
        args.muf_inclusion * float(replay_jacobians[d, inclusion].sum())
        for d in range(n_det)
    ])
    replay_gradient    = np.tensordot(residual, replay_jacobians,    axes=(0, 0))
    twofield_gradient  = np.tensordot(residual, twofield_jacobians,  axes=(0, 0))
    gradient_mask = np.isfinite(replay_gradient) & np.isfinite(twofield_gradient) & mask
    corr = correlation(replay_gradient, twofield_gradient, gradient_mask)

    truth = np.where(inclusion, args.muf_inclusion, 0.0)

    # --- Save arrays ---
    np.save(args.output_dir / "fluo_jacobians.npy",           replay_jacobians)
    np.save(args.output_dir / "fluo_twofield_jacobians.npy",  twofield_jacobians)
    np.save(args.output_dir / "fluo_replay_gradient.npy",     replay_gradient)
    np.save(args.output_dir / "fluo_twofield_gradient.npy",   twofield_gradient)

    # --- Plots ---
    saved_plots = False
    try:
        # Show excitation field + two representative emission adjoint fields
        field_det_indices = [0, n_det // 2] if n_det > 1 else [0]
        save_field_visualization(
            G_exc,
            [G_em_adjs[i] for i in field_det_indices],
            field_det_indices,
            args.output_dir / "fluo_fields.png",
        )
        save_gradient_comparison(
            replay_gradient, twofield_gradient, truth, gradient_mask, corr,
            args.output_dir / "fluo_comparison.png",
            args.robust_percentile,
        )
        saved_plots = True
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise

    # --- Summary ---
    print(f"mua          {args.mua} mm⁻¹")
    print(f"muaf         {args.muaf} mm⁻¹")
    print(f"mus          {args.mus} mm⁻¹")
    print(f"muf_inc      {args.muf_inclusion}")
    print(f"residual     {residual}")
    print(f"corr         {corr:.6f}")
    print(f"twofield x   {scale:.4e}")
    print(f"saved        {args.output_dir / 'fluo_jacobians.npy'}")
    print(f"saved        {args.output_dir / 'fluo_twofield_jacobians.npy'}")
    if saved_plots:
        print(f"saved        {args.output_dir / 'fluo_fields.png'}")
        print(f"saved        {args.output_dir / 'fluo_comparison.png'}")
    else:
        print("plots        skipped (install matplotlib or use the project venv)")


if __name__ == "__main__":
    main()
