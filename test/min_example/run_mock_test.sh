#!/bin/bash

set -e

cd "$(dirname "$0")"

python3 make_volume.py
python3 make_config.py --mua-inclusion 0.03 --session obs --output configs/obs.json
python3 make_config.py --mua-inclusion 0.01 --session fwd --output configs/case.json
python3 optimize.py --mock

test -f data/volume.npy
test -f configs/obs.json
test -f configs/case.json
test -f outputs/gradient.npy

echo "min_example mock test passed"
