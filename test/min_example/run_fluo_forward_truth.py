"""
Ground-truth fluorescence detector signal via an analog two-stage forward MC.

Unlike the replay/two-field Jacobians in run_fluo_comparison.py (both of which
are *linearizations* of d y_det / d muf[x]), this script samples the actual
nonlinear forward measurement: an ordinary excitation random walk that, at
each interaction, stochastically chooses between scattering (rate mus) and a
one-time wavelength-changing emission event (rate muf), continues as an
emission-wavelength photon with a fresh isotropic direction, and applies
microscopic Beer-Lambert absorption with mua before emission and muaf after.
This is implemented directly in the otFluoReplay-adjacent forward path of
mcx_core.cl (active whenever both muaf and muf volumes are supplied to a
plain forward run, i.e. outputtype != "fluo"); the photon's live two-stage
attenuated exit weight is returned per detected photon as result["fluoweight"].

Output files in --output-dir:
  fluo_forward_truth.npy      - (n_det,) MC-sampled per-detector signal
  fluo_forward_vs_linear.png  - bar comparison against the linearized replay
                                 estimate from run_fluo_comparison.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "pmcxcl"))
import pmcxcl  # type: ignore

sys.path.insert(0, str(ROOT))
from run_fluo_comparison import (  # noqa: E402
    NX, NY, NZ, SOURCE_POS, SOURCE_DIR, DETECTORS,
    make_volume, make_muf_vol, make_prop, _base_kwargs,
    run_fluo_replay,
)


def run_fluo_forward(vol, prop_exc, muaf_vol, muf_vol, nphoton, gpuid) -> dict:
    """Analog two-stage forward MC: real random-walk sample of the
    fluorescence measurement, not a linearized Jacobian."""
    return pmcxcl.run(
        **_base_kwargs(vol, prop_exc, nphoton, gpuid),
        muaf=muaf_vol, muf=muf_vol,
        srcpos=SOURCE_POS, srcdir=SOURCE_DIR,
        detpos=DETECTORS,
        issavedet=1, issaveseed=0, savedetflag="DP",
        outputtype="flux",
    )


def per_detector_signal(detp: np.ndarray, fluoweight: np.ndarray, n_det: int) -> np.ndarray:
    """Mean two-stage-attenuated exit weight per detector (0 for detectors
    with no hits). detp[0] holds the 1-indexed detector id (savedetflag='DP')."""
    detid = np.asarray(detp)[0].astype(int)
    signal = np.zeros(n_det, dtype=np.float64)
    for d in range(1, n_det + 1):
        hits = fluoweight[detid == d]
        if hits.size:
            signal[d - 1] = hits.mean()
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the ground-truth fluorescence detector signal via "
                    "an analog two-stage forward Monte Carlo, and sanity-check "
                    "it against the linearized replay Jacobian estimate."
    )
    parser.add_argument("--mua",  type=float, default=0.01)
    parser.add_argument("--muaf", type=float, default=0.02)
    parser.add_argument("--mus",  type=float, default=10.0)
    parser.add_argument("--g",    type=float, default=0.9)
    parser.add_argument("--n",    type=float, default=1.37)
    parser.add_argument("--muf-inclusion", type=float, default=0.1,
                        help="fluorophore concentration in inclusion (default 0.1)")
    parser.add_argument("--photons", type=int, default=2_000_000,
                        help="photons for the forward MC sample (default 2,000,000; "
                             "needs to be much larger than run_fluo_comparison.py's "
                             "since fluorescence detection is a rare event here)")
    parser.add_argument("--gpuid", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    vol = make_volume()
    inclusion = (vol == 2)
    prop_exc = make_prop(args.mua, args.mus, args.g, args.n)
    muaf_vol = np.full(vol.shape, args.muaf, dtype=np.float32, order="F")
    muf_vol = make_muf_vol(vol, args.muf_inclusion)
    n_det = len(DETECTORS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running analog forward fluorescence MC ({args.photons:,} photons) ...")
    fwd = run_fluo_forward(vol, prop_exc, muaf_vol, muf_vol, args.photons, args.gpuid)
    if "fluoweight" not in fwd or "detp" not in fwd:
        raise SystemExit("No fluoresced+detected photons — increase --photons or --muf-inclusion.")
    fluoweight = np.asarray(fwd["fluoweight"]).astype(np.float64)
    detp = fwd["detp"]
    n_detected = detp.shape[1]
    print(f"  Detected {n_detected:,} photons across {n_det} detectors.")

    # truth_signal[d] = mean two-stage-attenuated exit weight of photons that
    # landed at detector d - the forward MC estimate of the fluorescence
    # measurement there, in the same units MCX uses for a photon's exit weight.
    truth_signal = per_detector_signal(detp, fluoweight, n_det)

    # --- Sanity check against the linearized replay estimate ---
    print("Running excitation forward + replay pass for a linear-regime comparison ...")
    fwd_exc = pmcxcl.run(
        **_base_kwargs(vol, prop_exc, args.photons, args.gpuid),
        srcpos=SOURCE_POS, srcdir=SOURCE_DIR, detpos=DETECTORS,
        issavedet=1, issaveseed=1, savedetflag="DP",
        outputtype="flux",
    )
    seeds, detp_exc = fwd_exc["seeds"], fwd_exc["detp"]
    prop_muaf = np.array([0.0, args.muaf, args.muaf], dtype=np.float32)
    muaf_full = np.full(vol.shape, args.muaf, dtype=np.float32, order="F")
    muf_zero = np.zeros(vol.shape, dtype=np.float32, order="F")

    linear_signal = np.zeros(n_det, dtype=np.float64)
    for det_idx in range(n_det):
        J = run_fluo_replay(vol, prop_exc, muaf_full, muf_zero, prop_muaf,
                            seeds, detp_exc, det_idx + 1, args.gpuid)
        linear_signal[det_idx] = args.muf_inclusion * float(J[inclusion].sum())

    np.save(args.output_dir / "fluo_forward_truth.npy", truth_signal)

    print("\ndetector   forward-MC signal   linearized replay estimate")
    for d in range(n_det):
        print(f"  {d + 1:2d}      {truth_signal[d]: .6e}        {linear_signal[d]: .6e}")

    try:
        import matplotlib.pyplot as plt
        # forward-MC and linearized-replay signals live on unrelated absolute
        # scales (different units/normalization - see module docstring), so
        # they're plotted on separate twin axes rather than a shared one,
        # which would otherwise flatten the smaller series to invisibility.
        x = np.arange(1, n_det + 1)
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        c1, c2 = "tab:blue", "tab:orange"
        b1 = ax1.bar(x - width / 2, truth_signal, width, color=c1, label="forward MC (ground truth)")
        b2 = ax2.bar(x + width / 2, linear_signal, width, color=c2, label="linearized replay estimate")
        ax1.set_xlabel("detector")
        ax1.set_ylabel("forward MC signal", color=c1)
        ax2.set_ylabel("linearized replay signal", color=c2)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax2.tick_params(axis="y", labelcolor=c2)
        ax1.set_title(f"Forward MC vs. linearized replay (muf_inclusion={args.muf_inclusion})\n"
                      f"note: separate y-axes - the two methods use different absolute units")
        ax1.legend(handles=[b1, b2], loc="upper left")
        fig.tight_layout()
        fig.savefig(args.output_dir / "fluo_forward_vs_linear.png", dpi=160)
        plt.close(fig)
        print(f"\nsaved        {args.output_dir / 'fluo_forward_vs_linear.png'}")
    except ModuleNotFoundError:
        print("\nplots        skipped (install matplotlib or use the project venv)")

    print(f"saved        {args.output_dir / 'fluo_forward_truth.npy'}")


if __name__ == "__main__":
    main()
