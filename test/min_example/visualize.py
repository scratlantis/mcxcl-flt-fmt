from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float64)
    return (values - lo) / (hi - lo)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [20, 22, 28],
            [42, 157, 143],
            [231, 111, 81],
        ],
        dtype=np.uint8,
    )
    return palette[np.clip(labels.astype(np.int64), 0, len(palette) - 1)]


def gradient_to_rgb(gradient: np.ndarray) -> np.ndarray:
    scaled = normalize(gradient)
    blue = np.array([49, 104, 142], dtype=np.float64)
    white = np.array([245, 245, 240], dtype=np.float64)
    red = np.array([202, 73, 85], dtype=np.float64)
    rgb = np.empty(gradient.shape + (3,), dtype=np.float64)
    low = scaled <= 0.5
    rgb[low] = blue + (white - blue) * (scaled[low, None] * 2.0)
    rgb[~low] = white + (red - white) * ((scaled[~low, None] - 0.5) * 2.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def upscale(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def write_png(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("expected RGB image")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def save_fallback_png(volume_slice: np.ndarray, gradient_slice: np.ndarray, output: Path) -> None:
    left = upscale(labels_to_rgb(volume_slice.T[::-1]), 8)
    right = upscale(gradient_to_rgb(gradient_slice.T[::-1]), 8)
    gutter = np.full((left.shape[0], 16, 3), 245, dtype=np.uint8)
    image = np.concatenate([left, gutter, right], axis=1)
    write_png(output, image)


def save_matplotlib_png(volume_slice: np.ndarray, gradient_slice: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(volume_slice.T, origin="lower", cmap="viridis")
    axes[0].set_title("labels")
    axes[1].imshow(gradient_slice.T, origin="lower", cmap="coolwarm")
    axes[1].set_title("gradient")
    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize central volume and gradient slices.")
    parser.add_argument("--volume", type=Path, default=Path("data/volume.npy"))
    parser.add_argument("--gradient", type=Path, default=Path("outputs/gradient.npy"))
    parser.add_argument("--output", type=Path, default=Path("outputs/summary.png"))
    args = parser.parse_args()

    volume = np.load(args.volume)
    gradient = np.load(args.gradient)
    z = volume.shape[2] // 2
    volume_slice = volume[:, :, z]
    gradient_slice = gradient[:, :, z]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_matplotlib_png(volume_slice, gradient_slice, args.output)
        print(f"saved {args.output} with matplotlib")
    except ModuleNotFoundError:
        save_fallback_png(volume_slice, gradient_slice, args.output)
        print(f"saved {args.output} without matplotlib")


if __name__ == "__main__":
    main()
