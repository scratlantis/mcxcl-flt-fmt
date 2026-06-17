#!/bin/bash

set -e

cd "$(dirname "$0")"

PHOTONS=${PHOTONS:-200000}
MCX=${MCX:-../../bin/mcxcl}
PYTHON=${PYTHON:-.venv/bin/python}
read -r -a MCX_EXTRA_ARGS_ARRAY <<< "${MCX_EXTRA_ARGS:-}"

export TMPDIR=${TMPDIR:-/tmp/pocl-temp}
export POCL_CACHE_DIR=${POCL_CACHE_DIR:-/tmp/pocl-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-.matplotlib}
mkdir -p "$TMPDIR" "$POCL_CACHE_DIR" "$MPLCONFIGDIR"

"$PYTHON" make_volume.py
"$PYTHON" make_config.py --mua-inclusion 0.03 --session obs --photons "$PHOTONS" --output configs/obs.json
"$PYTHON" make_config.py --mua-inclusion 0.01 --session fwd --photons "$PHOTONS" --output configs/case.json
"$PYTHON" run_forward.py --config configs/obs.json --session obs --photons "$PHOTONS" --mcx "$MCX" "${MCX_EXTRA_ARGS_ARRAY[@]/#/--mcx-arg=}"
"$PYTHON" run_forward.py --config configs/case.json --session fwd --photons "$PHOTONS" --mcx "$MCX" "${MCX_EXTRA_ARGS_ARRAY[@]/#/--mcx-arg=}"
"$PYTHON" run_replay.py --config configs/case.json --seed outputs/fwd_detp.jdat --detectors 9 --mcx "$MCX" "${MCX_EXTRA_ARGS_ARRAY[@]/#/--mcx-arg=}"
"$PYTHON" optimize.py
"$PYTHON" visualize.py
"$PYTHON" run_twofield.py \
  --config configs/case.json \
  --obs-config configs/obs.json \
  --pred-detp outputs/fwd_detp.jdat \
  --obs-detp outputs/obs_detp.jdat \
  --total-photons "$PHOTONS" \
  --mcx "$MCX" \
  "${MCX_EXTRA_ARGS_ARRAY[@]/#/--mcx-arg=}"

echo "min_example local demo passed"
