"""
Verify gfluoweight accumulation in the otFluoReplay forward pass.

Experiment A: mua == muaf everywhere
  -> fluoweight[i] should equal replayweight[i] for every detected photon

Experiment B: muaf = 4 * mua everywhere
  -> fluoweight[i] should be systematically smaller than replayweight[i]
     (more absorption at emission wavelength means fewer photons survive)

Output: fluo_weight_histograms.png saved next to this script.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pmcxcl"))
import pmcxcl  # type: ignore

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
NX, NY, NZ = 60, 60, 60
vol = np.ones((NX, NY, NZ), dtype=np.uint8, order="F")

BASE_MUA = 0.005   # mm^-1

def run_experiment(label: str, mua: float, muaf: float):
    """Run forward sim; return (replayweight, fluoweight) per detected photon."""
    prop = np.array([
        [0.0,  0.0, 1.0, 1.00],
        [mua,  1.0, 0.0, 1.37],
    ], dtype=np.float32, order="F")

    muaf_vol = np.full((NX, NY, NZ), muaf, dtype=np.float32, order="F")

    cfg = {
        "nphoton": 2e7,
        "vol": vol,
        "prop": prop,
        "muaf": muaf_vol,
        "srcpos": [30, 30, 0],
        "srcdir": [0, 0, 1],
        "issrcfrom0": 1,
        "tstart": 0.0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "detpos": np.array([[30, 45, 0, 4]], dtype=np.float32),
        "issavedet": 1,
        "issaveseed": 0,
        "savedetflag": "DP",   # detid + partial paths
        "outputtype": "fluo",
        "isnormalized": 0,
        "isatomic": 0,
        "autopilot": 1,
        "gpuid": 1,
    }

    result = pmcxcl.run(**cfg)

    detp = result.get("detp")
    fluoweight = result.get("fluoweight")

    if detp is None:
        raise RuntimeError(f"[{label}] 'detp' missing from result. keys={list(result.keys())}")
    if fluoweight is None:
        raise RuntimeError(f"[{label}] 'fluoweight' missing from result — "
                           "check that gfluoweight is wired up correctly.")

    # replayweight = exp(-mua * total_path_in_medium_1)
    # detp row 0 = detid, row 1 = partial path in medium 1 (mm)
    plen_med1 = np.asarray(detp)[1, :]
    replayweight = np.exp(-mua * plen_med1)
    fluoweight = np.asarray(fluoweight)

    n = min(len(replayweight), len(fluoweight))
    replayweight, fluoweight = replayweight[:n], fluoweight[:n]

    print(f"[{label}] detected={n:,}  "
          f"rw mean={replayweight.mean():.4f}  fw mean={fluoweight.mean():.4f}  "
          f"fw/rw ratio mean={( fluoweight/(replayweight+1e-15)).mean():.4f}")

    return replayweight, fluoweight


# ---------------------------------------------------------------------------
# Run both experiments
# ---------------------------------------------------------------------------
rw_A, fw_A = run_experiment("A: mua==muaf", BASE_MUA, BASE_MUA)
rw_B, fw_B = run_experiment("B: muaf=4*mua", BASE_MUA, BASE_MUA * 4)

# ---------------------------------------------------------------------------
# Sanity check A: rw and fw should be identical within float precision
# ---------------------------------------------------------------------------
rel_err = np.abs(rw_A - fw_A) / (rw_A + 1e-15)
print(f"\n[A] max relative |rw-fw|/rw = {rel_err.max():.6f}  "
      f"(should be ~0 if muaf accumulation is correct)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
bins = np.linspace(0, 1, 60)

for row, (label, rw, fw) in enumerate([
    ("A: mua == muaf", rw_A, fw_A),
    ("B: muaf = 4·mua", rw_B, fw_B),
]):
    ax_rw = axes[row, 0]
    ax_fw = axes[row, 1]

    ax_rw.hist(rw, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax_rw.set_title(f"{label}\nreplayweight  exp(−mua·L)")
    ax_rw.set_xlabel("weight")
    ax_rw.set_ylabel("photon count")

    ax_fw.hist(fw, bins=bins, color="tomato", edgecolor="none", alpha=0.85)
    ax_fw.set_title(f"{label}\nfluoweight  exp(−muaf·L)")
    ax_fw.set_xlabel("weight")

fig.suptitle("Forward pass weight histograms: replayweight vs fluoweight", fontsize=12)
plt.tight_layout()

out = Path(__file__).with_name("fluo_weight_histograms.png")
fig.savefig(out, dpi=130)
print(f"\nSaved: {out}")
