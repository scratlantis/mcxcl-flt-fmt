"""
Test the fluorescence Jacobian computed by the otFluoReplay replay pass.

Geometry: homogeneous slab (5 x 5 x 40 voxels, 1 mm each), all medium 1.
Source: pencil beam at (2, 2, 0) pointing in +z. No scattering (mus = 0).
Detector: disk at z = 40, radius 3 mm.

With mus = 0 every photon travels straight from z = 0 to z = 40.  The
Jacobian column through the beam axis (x=2, y=2) has the closed-form:

    J[k] = N_det * exp(-mua * k) * exp(-muaf * (NZ - k))   k = 0 .. NZ-1

  where
    exp(-mua * k)       -- excitation weight at entry to voxel k (1 mm steps)
    exp(-muaf*(NZ-k))   -- emission transmittance from voxel-k entry to detector

The forward pass uses outputtype="flux" (no fluorescence kernel args) to
safely collect seeds and partial paths.  The replay pass adds muaf/muf and
uses outputtype="fluo" (otFluoReplay) to compute the Jacobian.

Output: fluo_jacobian_test.png saved next to this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pmcxcl"))
import pmcxcl  # type: ignore
import pmcx    # utility functions: detweight, meanpath, etc.

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
NX, NY, NZ = 5, 5, 40
mua  = 0.01   # mm^-1  excitation absorption
muaf = 0.02   # mm^-1  emission absorption

vol = np.ones((NX, NY, NZ), dtype=np.uint8, order="F")

prop = np.array([
    [0.0, 0.0, 1.0, 1.0],   # medium 0: void
    [mua, 0.0, 1.0, 1.0],   # medium 1: absorber, no scatter
], dtype=np.float32, order="F")

# ---------------------------------------------------------------------------
# Forward pass — collect seeds and partial paths only.
# No muaf/muf here so the kernel has no HAS_MUAF and no null GPU buffers.
# ---------------------------------------------------------------------------
print("Running forward pass ...")
fwd = pmcxcl.run(
    vol=vol,
    prop=prop,
    nphoton=5e5,
    srcpos=[2, 2, 0],
    srcdir=[0, 0, 1],
    issrcfrom0=1,
    tstart=0.0,
    tend=5e-9,
    tstep=5e-9,
    detpos=np.array([[2, 2, NZ, 3]], dtype=np.float32),
    issavedet=1,
    issaveseed=1,
    savedetflag="DP",
    outputtype="flux",
    isnormalized=0,
    isatomic=0,
    autopilot=1,
    gpuid=1,
)

seed  = fwd["seeds"]  # (RAND_WORD_LEN*4, N_det) uint8
detp  = fwd["detp"]   # (hostdetreclen, N_det) float32
N_det = seed.shape[1]
print(f"  Detected photons: {N_det:,}")

if N_det == 0:
    raise SystemExit("No photons detected — check geometry / detector position.")

# Use pmcx.detweight to cross-check the per-photon excitation weight.
# pmcx.detweight expects a dict with ppath shaped (N_photons, N_media-1);
# detp[0,:] = detid, detp[1:,:] = partial paths -> transpose to (N_det, N_media-1).
detp_arr = np.asarray(detp)
detp_dict = {"ppath": detp_arr[1:, :].T}   # (N_det, N_media-1)
rw = pmcx.detweight(detp_dict, prop)
print(f"  Excitation replayweight: mean={rw.mean():.6f}  "
      f"expected={np.exp(-mua * NZ):.6f}  (all photons same straight path)")

# ---------------------------------------------------------------------------
# Replay pass — fluorescence Jacobian.
# muaf volume and muf (zeros, needed to keep gmuf a valid GPU buffer even
# though the replay branch never reads it) are added here only.
# prop_muaf tells mcx_replayinit to compute fluoweight from partial paths.
# ---------------------------------------------------------------------------
muaf_vol = np.full((NX, NY, NZ), muaf, dtype=np.float32, order="F")
muf_vol  = np.zeros((NX, NY, NZ), dtype=np.float32, order="F")
prop_muaf = np.array([0.0, muaf], dtype=np.float32)

# --- Sanity check: mua Jacobian replay ---
# J_mua[k] = N_det * replayweight * pathlen = N_det * exp(-mua*40) * 1.0 (constant)
print("Running mua Jacobian replay (sanity check) ...")
jmua_rep = pmcxcl.run(
    vol=vol, prop=prop, nphoton=N_det, seed=seed, detphotons=detp,
    srcpos=[2, 2, 0], srcdir=[0, 0, 1], issrcfrom0=1,
    tstart=0.0, tend=5e-9, tstep=5e-9,
    detpos=np.array([[2, 2, NZ, 3]], dtype=np.float32),
    issavedet=0, issaveseed=0, outputtype="jacobian",
    isnormalized=0, isatomic=1, autopilot=1, gpuid=1,
)
Jmua = np.squeeze(jmua_rep["flux"])
J_mua_col = Jmua[2, 2, :]
J_mua_ref = N_det * np.exp(-mua * NZ)   # constant over all voxels
print(f"  J_mua[2,2,:5] = {J_mua_col[:5]}")
print(f"  J_mua_ref     = {J_mua_ref:.4f}")
print(f"  scale ratio J_mua_ref / J_mua_col.mean() = {J_mua_ref / J_mua_col.mean():.4f}")

print("Running replay pass ...")
replay = pmcxcl.run(
    vol=vol,
    prop=prop,
    muaf=muaf_vol,
    muf=muf_vol,
    nphoton=N_det,
    seed=seed,
    detphotons=detp,
    prop_muaf=prop_muaf,
    srcpos=[2, 2, 0],
    srcdir=[0, 0, 1],
    issrcfrom0=1,
    tstart=0.0,
    tend=5e-9,
    tstep=5e-9,
    detpos=np.array([[2, 2, NZ, 3]], dtype=np.float32),
    issavedet=0,
    issaveseed=0,
    outputtype="fluo",    # otFluoReplay -> fluorescence Jacobian
    isnormalized=0,
    isatomic=1,
    autopilot=1,
    gpuid=1,
)

# flux shape: (NX, NY, NZ, 1) in F-style -> squeeze time-gate dim
J_raw = replay["flux"]
print(f"  flux shape: {J_raw.shape}  total deposited: {J_raw.sum():.4f}  max: {J_raw.max():.4f}")
J = np.squeeze(J_raw)   # (NX, NY, NZ)
print(f"  nonzero voxels: {np.count_nonzero(J)}")
print(f"  J[2,2,:5] = {J[2,2,:5]}")

# Beam axis column: photon traverses voxels (2, 2, 0..NZ-1)
J_col    = J[2, 2, :].astype(np.float64)
J_mua_col = J_mua_col.astype(np.float64)

# ---------------------------------------------------------------------------
# Analytical reference — ratio cancels the common MCX scale factor
#
# J_fluo[k] = C * exp(-mua*k)   * exp(-muaf*(NZ-k))
# J_mua[k]  = C * exp(-mua*NZ)  (constant, replayweight same for all photons)
#
#   => J_fluo[k] / J_mua[k] = exp((mua - muaf) * (NZ - k))
# ---------------------------------------------------------------------------
k = np.arange(NZ, dtype=np.float64)
ratio     = J_col / (J_mua_col + 1e-30)
ratio_ref = np.exp((mua - muaf) * (NZ - k))

rel_err = np.abs(ratio - ratio_ref) / (ratio_ref + 1e-30)
max_rel_err = rel_err.max()
print(f"  Max relative error (ratio vs analytical): {max_rel_err:.4f}")

rtol = 0.10   # 10%: generous for 500k-photon MC noise on a ratio
if max_rel_err > rtol:
    print(f"  WARNING: max relative error {max_rel_err:.4f} exceeds tolerance {rtol}")
else:
    print(f"  PASS (tolerance {rtol})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(k, ratio_ref, "k--", label="analytical ratio", linewidth=1.5)
ax.plot(k, ratio, "o", markersize=3, label="MCX  J_fluo / J_mua", alpha=0.7)
ax.set_xlabel("voxel index k  (z-axis)")
ax.set_ylabel("J_fluo[k] / J_mua[k]")
ax.set_title("Fluorescence Jacobian ratio (cancels MCX scale)")
ax.legend()

ax = axes[1]
ax.semilogy(k, rel_err + 1e-8)
ax.axhline(rtol, color="red", linestyle="--", label=f"tolerance {rtol}")
ax.set_xlabel("voxel index k")
ax.set_ylabel("|ratio - ratio_ref| / ratio_ref")
ax.set_title("Relative error of ratio")
ax.legend()

fig.suptitle(
    f"Fluorescence Jacobian test  "
    f"(mua={mua} mm⁻¹, muaf={muaf} mm⁻¹, NZ={NZ}, N_det={N_det:,})",
    fontsize=10,
)
plt.tight_layout()

out = Path(__file__).with_name("fluo_jacobian_test.png")
fig.savefig(out, dpi=130)
print(f"Saved: {out}")
