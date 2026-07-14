#!/usr/bin/env python3
"""Regression test: replaydet=-1 matches separate fluorescence replays."""

from __future__ import annotations

import numpy as np

import pmcxcl


def main() -> None:
    shape = (20, 20, 20)
    volume = np.ones(shape, dtype=np.uint8, order="F")
    prop = np.asfortranarray(
        np.array([[0.0, 0.0, 1.0, 1.0], [0.005, 1.0, 0.9, 1.37]], dtype=np.float32)
    )
    detectors = np.asfortranarray(
        np.array([[6.0, 6.0, 20.0, 3.0], [14.0, 6.0, 20.0, 3.0],
                  [6.0, 14.0, 20.0, 3.0], [14.0, 14.0, 20.0, 3.0]], dtype=np.float32)
    )
    source = [10.0, 10.0, 0.0]
    forward = pmcxcl.run(
        vol=volume, prop=prop, nphoton=100_000, tstart=0.0, tend=5e-9, tstep=5e-9,
        issrcfrom0=1, isnormalized=0, isatomic=1, autopilot=1, gpuid=1,
        srcpos=source, srcdir=[0.0, 0.0, 1.0], detpos=detectors,
        issavedet=1, issaveseed=1, savedetflag="DP", issave2pt=False,
        outputtype="flux", seed=12345,
    )
    seeds, detp = forward["seeds"], forward["detp"]
    assert seeds.shape[1] > 0

    muaf = np.full(shape, 0.01, dtype=np.float32, order="F")
    muf = np.zeros(shape, dtype=np.float32, order="F")
    common = dict(
        vol=volume, prop=prop, nphoton=int(seeds.shape[1]),
        tstart=0.0, tend=5e-9, tstep=5e-9, issrcfrom0=1,
        isnormalized=0, isatomic=1, autopilot=1, gpuid=1,
        muaf=muaf, muf=muf, prop_muaf=np.array([0.0, 0.01], dtype=np.float32),
        seed=seeds, detphotons=detp, srcpos=source, srcdir=[0.0, 0.0, 1.0],
        detpos=detectors, issavedet=0, issaveseed=0, outputtype="fluo",
    )
    combined = np.asarray(pmcxcl.run(**common, replaydet=-1)["flux"])
    assert combined.shape == (*shape, 1, len(detectors))

    for detector in range(1, len(detectors) + 1):
        separate = np.squeeze(np.asarray(pmcxcl.run(**common, replaydet=detector)["flux"]))
        np.testing.assert_allclose(
            combined[:, :, :, 0, detector - 1], separate, rtol=1e-5, atol=1e-6
        )

    print(f"PASS: {len(detectors)} detector volumes match separate replay calls")


if __name__ == "__main__":
    main()
