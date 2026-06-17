from __future__ import annotations

import json
import struct
import base64
import zlib
from pathlib import Path

import numpy as np


def load_mch(path: Path) -> tuple[np.ndarray, dict]:
    """Load a legacy MCX `.mch` history file."""
    with path.open("rb") as handle:
        if handle.read(4) != b"MCXH":
            raise ValueError(f"{path} is not an MCX history file")
        version, media_count, det_count, record_count, photons, detected, saved = struct.unpack(
            "<7I", handle.read(28)
        )
        unit_mm = struct.unpack("<f", handle.read(4))[0]
        seed_bytes = struct.unpack("<I", handle.read(4))[0]
        normalizer = struct.unpack("<f", handle.read(4))[0]
        respin = struct.unpack("<i", handle.read(4))[0]
        src_num = struct.unpack("<I", handle.read(4))[0]
        savedetflag = struct.unpack("<I", handle.read(4))[0]
        total_source = struct.unpack("<I", handle.read(4))[0]
        handle.seek(256)
        values = np.frombuffer(handle.read(saved * record_count * 4), dtype="<f4")

    data = values.reshape((saved, record_count)) if saved else np.empty((0, record_count))
    header = {
        "version": version,
        "media_count": media_count,
        "det_count": det_count,
        "record_count": record_count,
        "photons": photons,
        "detected": detected,
        "saved": saved,
        "unit_mm": unit_mm,
        "seed_bytes": seed_bytes,
        "normalizer": normalizer,
        "respin": respin,
        "src_num": src_num,
        "savedetflag": savedetflag,
        "total_source": total_source,
    }
    return data, header


def measurements_from_mch(path: Path, mua_by_label: np.ndarray, det_count: int = 4) -> np.ndarray:
    data, header = load_mch(path)
    if data.size == 0:
        return np.zeros(det_count, dtype=np.float64)

    media_count = header["media_count"]
    det_ids = data[:, 0].astype(np.int64)
    ppath_start = 1 + media_count
    ppath_stop = ppath_start + media_count
    if data.shape[1] < ppath_stop:
        raise ValueError("detected photon records do not include partial path lengths")

    ppaths = data[:, ppath_start:ppath_stop]
    weights = np.exp(-(ppaths * mua_by_label[:media_count]).sum(axis=1))
    measurements = np.zeros(det_count, dtype=np.float64)
    for det_idx in range(1, det_count + 1):
        measurements[det_idx - 1] = weights[det_ids == det_idx].sum()
    return measurements


def decode_jdata_array(node: dict) -> np.ndarray:
    type_map = {
        "single": "<f4",
        "double": "<f8",
        "uint32": "<u4",
        "int32": "<i4",
        "uint16": "<u2",
        "int16": "<i2",
        "uint8": "u1",
        "int8": "i1",
    }
    dtype = type_map[node["_ArrayType_"]]
    raw = base64.b64decode(node["_ArrayZipData_"])
    if node.get("_ArrayZipType_") == "zlib":
        raw = zlib.decompress(raw)
    array = np.frombuffer(raw, dtype=np.dtype(dtype))
    return array.reshape(tuple(node["_ArraySize_"]))


def measurements_from_jdat(path: Path, mua_by_label: np.ndarray, det_count: int = 4) -> np.ndarray:
    payload = json.loads(path.read_text())["MCXData"]
    photon_data = payload["PhotonData"]
    det_ids = decode_jdata_array(photon_data["detid"]).reshape(-1).astype(np.int64)
    ppaths = decode_jdata_array(photon_data["ppath"]).astype(np.float64)
    weights = np.exp(-(ppaths * mua_by_label[1 : ppaths.shape[1] + 1]).sum(axis=1))
    normalizer = float(payload["Info"].get("Normalizer", 1.0))

    measurements = np.zeros(det_count, dtype=np.float64)
    for det_idx in range(1, det_count + 1):
        measurements[det_idx - 1] = weights[det_ids == det_idx].sum() * normalizer
    return measurements


def load_jnii_array(path: Path) -> np.ndarray:
    try:
        from jdata import loadjd

        payload = loadjd(str(path))
    except ImportError:
        payload = json.loads(path.read_text())

    if "NIFTIData" in payload:
        data = payload["NIFTIData"]
        if isinstance(data, dict) and "_ArrayZipData_" in data:
            return decode_jdata_array(data).astype(np.float64)
        return np.asarray(data, dtype=np.float64)
    if "MCXData" in payload:
        data = payload["MCXData"]
        if isinstance(data, dict) and "_ArrayZipData_" in data:
            return decode_jdata_array(data).astype(np.float64)
        return np.asarray(data, dtype=np.float64)
    raise KeyError(f"could not find NIFTIData or MCXData in {path}")


def mock_measurements(mua_inclusion: float) -> np.ndarray:
    baseline = np.array([1.00, 0.92, 0.92, 0.84], dtype=np.float64)
    sensitivity = np.array([4.5, 5.0, 5.0, 5.8], dtype=np.float64)
    return baseline * np.exp(-sensitivity * mua_inclusion)


def mock_jacobians(volume: np.ndarray) -> np.ndarray:
    inclusion = volume == 2
    jacobians = np.zeros((4,) + volume.shape, dtype=np.float64)
    centers = np.array([[15, 15, 1], [25, 15, 1], [15, 25, 1], [25, 25, 1]], dtype=np.float64)
    grid = np.indices(volume.shape, dtype=np.float64)
    for idx, center in enumerate(centers):
        dist2 = sum((grid[axis] - center[axis]) ** 2 for axis in range(3))
        jacobians[idx] = -np.exp(-dist2 / 450.0) * inclusion
    return jacobians
