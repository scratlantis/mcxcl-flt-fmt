# Minimal MCXCL Inverse Test

This directory contains a small inverse-problem regression/demo for the MCXCL
fork: one pencil source, nine opposite-face detector ROIs, synthetic
observations, replay Jacobians, gradient accumulation, and one optical-property
update.

Generated files are intentionally ignored:

* `configs/`
* `data/`
* `outputs/`
* `.matplotlib/`
* `.venv/`

## Setup

```bash
cd test/min_example
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Forward/replay runs need an `mcxcl` executable on `PATH`, or pass a binary path
with `--mcx`.

From the repository root, a local command-line build can be made with:

```bash
cmake -S src -B build -DBUILD_MEX=OFF
cmake --build build --parallel 4
```

On POCL CPU OpenCL devices, set writable compiler scratch/cache directories if
kernel compilation fails with `mkstemp() failed`:

```bash
export TMPDIR=/tmp/pocl-temp
export POCL_CACHE_DIR=/tmp/pocl-cache
mkdir -p "$TMPDIR" "$POCL_CACHE_DIR"
```

## Smoke Test Without MCX

This generates the volume and configs, then runs the optimizer in deterministic
mock mode:

```bash
./run_mock_test.sh
```

## MCX Run

```bash
python3 make_volume.py
python3 make_config.py --mua-inclusion 0.03 --session obs --output configs/obs.json
python3 make_config.py --mua-inclusion 0.01 --session fwd --output configs/case.json
python3 run_forward.py --config configs/obs.json --session obs --photons 1000000
python3 run_forward.py --config configs/case.json --session fwd --photons 1000000
python3 run_replay.py --config configs/case.json --seed outputs/fwd_detp.jdat --detectors 9
python3 optimize.py
python3 visualize.py
```

For a one-command local run against the binary built at `../../bin/mcxcl`:

```bash
./run_local_demo.sh
```

Note: MCX replay uses `-Y/--replaydet` for detector selection. `-P` is reserved for shape injection in current MCX builds.

## Scattering Reconstruction

To make the synthetic target differ in scattering instead of absorption, keep `mua`
fixed and perturb `mus`:

```bash
python3 make_config.py --mua-inclusion 0.01 --mus-inclusion 12.0 --session obs --output configs/obs.json
python3 make_config.py --mua-inclusion 0.01 --mus-inclusion 10.0 --session fwd --output configs/case.json
```

Replay scattering needs both weighted pathlengths and scattering counts:

```bash
python3 run_replay.py --config configs/case.json --seed outputs/fwd_detp.jdat --detectors 9 --property mus --mcx mcxcl
```

This writes the usual `outputs/J_det_*.jnii` plus `outputs/P_det_*.jnii`.
The replay scattering Jacobian is computed as `J_mus = P / mus - J`, where
`J` is the replay weighted pathlength map and `P` is the replay weighted
scattering-count map. Then run one scattering update with:

```bash
python3 optimize.py --property mus
```

## Two-Field Adjoint Gradient

After the forward and replay commands above have produced `outputs/fwd_detp.jdat`,
`outputs/obs_detp.jdat`, and `outputs/J_det_*.jnii`, run:

```bash
python3 run_twofield.py --config configs/case.json --obs-config configs/obs.json --photons 1000000
```

This writes:

* `configs/source.json`
* `configs/adj_det_1.json` through `configs/adj_det_9.json`
* `outputs/G_source.jnii`
* `outputs/G_det_1.jnii` through `outputs/G_det_9.jnii`
* `outputs/twofield_jacobians.npy`
* `outputs/twofield_gradient.npy`
* `outputs/replay_mua_gradient_from_residual.npy`
* `outputs/twofield_comparison.png`
* `outputs/twofield_fields.png`

To regenerate only the figures from saved fields and gradients:

```bash
python3 visualize_twofield.py
```

Use `--skip-mcx` with `run_twofield.py` to rebuild the NumPy gradients and plots from existing `G_source.jnii` and `G_det_*.jnii` files.

By default, `run_twofield.py` uses `J_i = +G_source * G_det_i` so the saved
two-field gradient matches the sign convention of MCX replay output in this
demo. Use `--twofield-sign -1` to recover the classical absorption derivative
sign convention `J_i = -G_source * G_det_i`.

For scattering, run the same comparison with:

```bash
python3 run_twofield.py --config configs/case.json --obs-config configs/obs.json --property mus --photons 1000000
```

The two-field scattering estimate uses the diffusion-style
`grad(G_source) dot grad(G_det) / (3 * (1-g) * mus^2)` form and compares it
against the replay `P / mus - J` Jacobian.

For equal replay/two-field photon budgets, use:

```bash
python3 run_equal_budget_comparison.py --photons 1000000 --mcx mcxcl
```

Replay uses one `fwd` simulation at the requested photon count. Two-field splits
the same total count across `G_source` plus the nine detector-adjoint fields.
For example, `--photons 1000000` runs ten two-field MCX jobs at 100000 photons
each.

For an equal-budget scattering comparison:

```bash
python3 run_equal_budget_comparison.py --photons 1000000 --property mus --obs-mua-inclusion 0.01 --fwd-mua-inclusion 0.01 --obs-mus-inclusion 12.0 --fwd-mus-inclusion 10.0 --mcx mcxcl
```
