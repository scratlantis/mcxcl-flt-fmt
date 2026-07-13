# Replay and Two-Field Jacobian Construction Report

This report describes how the mini-example scripts build Jacobians for three parameter cases:

- absorption coefficient, `mu_a`
- scattering coefficient, `mu_s`
- fluorescent density / fluorophore concentration, `mu_f`

The main comparison drivers are:

- `run_equal_budget_comparison.py` for `mu_a` and `mu_s`
- `run_replay.py` for MCX replay Jacobian files
- `run_twofield.py` for two-field adjoint Jacobians
- `run_fluo_comparison.py` for fluorescence density Jacobians

## Shared Setup

The minimal example uses a `40 x 40 x 40` volume with:

- label `1`: background tissue
- label `2`: spherical inclusion
- nine detectors on the exit face
- one pencil source on the entrance face

For `mu_a` and `mu_s`, `run_equal_budget_comparison.py` orchestrates the workflow:

1. Build the volume with `make_volume.py`.
2. Write an observed config, `configs/obs.json`.
3. Write a forward/current config, `configs/case.json`.
4. Run forward MCX for the observed config.
5. Run forward MCX for the current config.
6. Run replay Jacobians from the current forward detected photons.
7. Run two-field adjoint Jacobians and compare them to replay.

The observed and current configs differ in the inclusion optical property. The resulting measurement residual is:

```text
residual[d] = y_pred[d] - y_obs[d]
```

The final gradient map is built from detector Jacobians as:

```text
gradient[x] = sum_d residual[d] * J[d, x]
```

where:

- `d` indexes detector
- `x` indexes voxel
- `J[d, x]` is the Jacobian for detector `d` at voxel `x`

## Case 1: Absorption Coefficient `mu_a`

### Replay Jacobian

The replay method uses the detected photon seeds and partial path information from the current forward simulation:

```text
outputs/fwd_detp.jdat
```

`run_replay.py` calls MCX once per detector with:

```text
-E outputs/fwd_detp.jdat
-O J
-Y detector_index
```

For absorption, MCX output type `J` is the replay absorption/pathlength Jacobian. The script writes one file per detector:

```text
outputs/J_det_1.jnii
outputs/J_det_2.jnii
...
outputs/J_det_9.jnii
```

Conceptually, for detector `d`, the replay absorption Jacobian at voxel `x` is accumulated from detected photons assigned to that detector:

```text
J_replay_mu_a[d, x] = replay accumulation of photon pathlength/sensitivity in voxel x
```

In the Python comparison code, these per-detector files are loaded as:

```text
replay_jacobians = load_replay_jacobians(output_dir, det_count, "J")
```

### Two-Field Jacobian

The two-field method first computes:

- `G_source[x]`: forward/source fluence field
- `G_det_d[x]`: adjoint detector fluence field for detector `d`

`run_twofield.py` writes source and detector-adjoint MCX configs:

```text
configs/source.json
configs/adj_det_1.json
configs/adj_det_2.json
...
configs/adj_det_9.json
```

Then it runs MCX to produce:

```text
outputs/G_source.jnii
outputs/G_det_1.jnii
outputs/G_det_2.jnii
...
outputs/G_det_9.jnii
```

For absorption, the two-field Jacobian is built by voxelwise multiplication:

```text
J_twofield_mu_a[d, x] = sign * G_source[x] * G_det_d[x]
```

In code:

```python
jacobians = np.stack(
    [args.twofield_sign * source * detector for detector in detector_fields],
    axis=0,
)
```

By default, `twofield_sign = 1.0`.

### Saved Outputs

For `mu_a`, `run_twofield.py` saves:

```text
outputs/twofield_jacobians.npy
outputs/replay_mua_jacobians.npy
outputs/twofield_gradient.npy
outputs/replay_mua_gradient_from_residual.npy
outputs/twofield_difference_normalized.npy
outputs/twofield_comparison_mask.npy
outputs/twofield_comparison.png
outputs/twofield_fields.png
```

## Case 2: Scattering Coefficient `mu_s`

### Replay Jacobian

For scattering, the replay method needs two MCX replay outputs:

```text
-O J
-O P
```

`run_replay.py` handles this when called with:

```text
--property mus
```

It writes:

```text
outputs/J_det_1.jnii ... outputs/J_det_9.jnii
outputs/P_det_1.jnii ... outputs/P_det_9.jnii
```

In `run_twofield.py`, the scattering replay components are loaded as:

```python
pathlength = load_replay_jacobians(output_dir, det_count, "J")
scat_counts = load_replay_jacobians(output_dir, det_count, "P")
mus = scattering_from_volume(volume_path, config)
scat_over_mus = scat_counts / mus
jacobians = scat_over_mus - pathlength
```

So the replay scattering Jacobian is:

```text
J_replay_mu_s[d, x] = P[d, x] / mu_s[x] - J[d, x]
```

where:

- `J[d, x]` is the replay pathlength/absorption-style term
- `P[d, x]` is the replay scattering-count term
- `mu_s[x]` is the scattering coefficient at voxel `x`

This is the convention used by both `run_twofield.py` and `optimize.py`.

### Two-Field Jacobian

The two-field scattering Jacobian uses the gradients of the source and detector adjoint fluence fields.

First, `run_twofield.py` computes the voxelwise gradient dot product:

```python
def gradient_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gradients_a = np.gradient(a)
    gradients_b = np.gradient(b)
    return sum(ga * gb for ga, gb in zip(gradients_a, gradients_b))
```

Then it computes a diffusion-style factor from each voxel's optical properties:

```text
factor[x] = 1 / (3 * (1 - g[x]) * mu_s[x]^2)
```

In code, this is built by `mus_adjoint_factor(...)` from the volume labels and `Domain.Media` entries.

The raw two-field scattering Jacobian is:

```text
J_twofield_mu_s_raw[d, x]
    = sign * dot(grad G_source[x], grad G_det_d[x])
      / (3 * (1 - g[x]) * mu_s[x]^2)
```

In code:

```python
factor = mus_adjoint_factor(args.volume, base_config_for_property)
jacobians = np.stack(
    [
        args.twofield_sign * gradient_dot(source, detector) * factor
        for detector in detector_fields
    ],
    axis=0,
)
```

### Optional Calibration

For `mu_s`, the script calibrates the raw two-field Jacobian to replay by default:

```python
twofield_scale = least_squares_scale(jacobians, replay_jacobians, mask[None, ...])
jacobians = jacobians * twofield_scale
```

So the saved two-field scattering Jacobian is:

```text
J_twofield_mu_s[d, x] = twofield_scale * J_twofield_mu_s_raw[d, x]
```

This calibration can be disabled with:

```text
--no-calibrate-twofield
```

### Masking

For `mu_s`, the comparison mask excludes shallow voxels by default:

```text
mask_z_min = 3
```

That means comparison and calibration use voxels with:

```text
z >= 3
```

### Saved Outputs

For `mu_s`, `run_twofield.py` saves:

```text
outputs/twofield_mus_jacobians.npy
outputs/replay_mus_jacobians.npy
outputs/twofield_mus_gradient.npy
outputs/replay_mus_gradient_from_residual.npy
outputs/twofield_mus_difference_normalized.npy
outputs/twofield_mus_comparison_mask.npy
outputs/twofield_mus_comparison.png
outputs/twofield_fields.png
outputs/mus_jacobian_diagnostics.png
outputs/mus_jacobian_diagnostics.json
```

## Case 3: Fluorescent Density `mu_f`

The fluorescence comparison is implemented separately in `run_fluo_comparison.py`.

Here the parameter of interest is fluorescent density / fluorophore concentration:

```text
mu_f[x]
```

The script compares:

```text
d y_det / d mu_f[x]
```

The optical properties are split into excitation and emission wavelengths:

- excitation absorption: `mu_a`
- emission absorption: `mu_af`
- scattering: `mu_s`
- anisotropy: `g`
- refractive index: `n`

### Replay Jacobian

The fluorescence replay path uses an excitation forward pass first:

```python
fwd = run_forward(vol, prop_exc, photons, gpuid)
seeds = fwd["seeds"]
detp = fwd["detp"]
G_exc = fwd["flux"]
```

The forward pass saves detected photon seeds and detector partial paths.

Then the script performs one fluorescence replay pass per detector:

```python
replay_jacobians[det_idx] = run_fluo_replay(
    vol,
    prop_exc,
    muaf_vol,
    muf_vol,
    prop_muaf,
    seeds,
    detp,
    det_idx + 1,
    gpuid,
)
```

Inside `run_fluo_replay`, pmcxcl is called with:

```text
outputtype = "fluo"
replaydet = detector_index
muaf = muaf_vol
muf = muf_vol
prop_muaf = prop_muaf
seed = seeds
detphotons = detp
```

Conceptually, this asks the MCX `otFluoReplay` kernel to compute:

```text
J_replay_mu_f[d, x] = d y_d / d mu_f[x]
```

The replay kernel combines:

- excitation photon history from the source to voxel `x`
- emission attenuation using `mu_af`
- detector selection through `replaydet`

In the fluorescence comparison script, `muf_vol` is set to zero for the replay Jacobian call:

```python
muf_vol = np.zeros(vol.shape, dtype=np.float32, order="F")
```

This is because the replay output is the derivative with respect to `mu_f`; the actual test fluorescence density is used later to construct a synthetic residual.

### Two-Field Jacobian

The fluorescence two-field method computes:

- `G_exc[x]`: excitation fluence from the real source using excitation optical properties
- `G_em_adj_d[x]`: emission adjoint fluence from detector `d` using emission optical properties

The emission adjoint run uses each detector as an isotropic source:

```python
result = pmcxcl.run(
    srcpos=detector_position,
    srctype="isotropic",
    outputtype="flux",
    prop=prop_em,
)
```

The raw two-field fluorescence Jacobian is:

```text
J_twofield_mu_f_raw[d, x] = G_exc[x] * G_em_adj_d[x]
```

In code:

```python
twofield_jacobians_raw = np.stack(
    [G_exc * G_adj for G_adj in G_em_adjs],
    axis=0,
)
```

Then the script calibrates the two-field scale to replay by least squares:

```python
mask = np.ones((NX, NY, NZ), dtype=bool)
scale = least_squares_scale(
    twofield_jacobians_raw.ravel(),
    replay_jacobians.ravel(),
    np.broadcast_to(mask, (n_det, NX, NY, NZ)).ravel(),
)
twofield_jacobians = twofield_jacobians_raw * scale
```

The current fluorescence comparison uses an all-voxel mask for this calibration.

So the saved two-field fluorescence Jacobian is:

```text
J_twofield_mu_f[d, x]
    = scale * G_exc[x] * G_em_adj_d[x]
```

### Fluorescence Gradient

The script creates a synthetic fluorescence residual using the replay Jacobian over the known fluorescent inclusion:

```python
residual[d] = muf_inclusion * sum_x_in_inclusion J_replay_mu_f[d, x]
```

Then it builds replay and two-field gradient maps:

```text
gradient_replay[x] = sum_d residual[d] * J_replay_mu_f[d, x]
gradient_twofield[x] = sum_d residual[d] * J_twofield_mu_f[d, x]
```

### Saved Outputs

For fluorescence density, `run_fluo_comparison.py` saves:

```text
outputs/fluo_jacobians.npy
outputs/fluo_twofield_jacobians.npy
outputs/fluo_replay_gradient.npy
outputs/fluo_twofield_gradient.npy
outputs/fluo_fields.png
outputs/fluo_comparison.png
```

## Summary Table

| Parameter | Replay construction | Two-field construction |
| --- | --- | --- |
| `mu_a` | MCX replay `-O J` per detector from detected photon seeds. | `G_source * G_detector_adjoint`. |
| `mu_s` | MCX replay `P / mu_s - J` per detector. | `dot(grad G_source, grad G_detector_adjoint) / (3 * (1 - g) * mu_s^2)`, then optionally least-squares scaled to replay. |
| `mu_f` | pmcxcl fluorescence replay with `outputtype="fluo"`, `muaf`, `muf`, `prop_muaf`, seeds, and detected photon paths. | `G_excitation_source * G_emission_detector_adjoint`, then least-squares scaled to replay. |

## Important Conventions

- `mu_a` and `mu_s` comparison use `run_equal_budget_comparison.py`.
- fluorescence density comparison uses `run_fluo_comparison.py`.
- Replay Jacobians are detector-resolved arrays with shape:

```text
(n_detectors, NX, NY, NZ)
```

- Final gradient maps are voxelwise detector-weighted sums:

```text
gradient[x] = sum_d residual[d] * J[d, x]
```

- The two-field `mu_a` and `mu_f` forms are fluence products.
- The two-field `mu_s` form is a gradient-dot-gradient expression with a diffusion coefficient derivative factor.
- The `mu_s` and `mu_f` two-field outputs are scaled to replay in the current scripts, while `mu_a` is not scaled by least squares.
