#!/usr/bin/env python3
"""Simple Tk slice viewer for the Digimouse JData/JNIfTI volumes."""

from __future__ import annotations

import argparse
import array
import base64
import json
import math
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
import zlib


DTYPE_INFO = {
    "uint8": ("B", 1),
    "uchar": ("B", 1),
    "byte": ("B", 1),
    "single": ("f", 4),
    "float32": ("f", 4),
    "float": ("f", 4),
}


def _regex_value(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text, re.S)
    if not match:
        raise ValueError(f"Could not find {key!r}")
    return match.group(1)


def _regex_array(text: str, key: str) -> list[int]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[([^\]]+)\]', text, re.S)
    if not match:
        raise ValueError(f"Could not find {key!r}")
    return [int(float(part.strip())) for part in match.group(1).split(",")]


def _load_json_or_relaxed(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        return {}, text


def load_volume(path: Path):
    obj, text = _load_json_or_relaxed(path)

    if "NIFTIData" in obj:
        container = obj["NIFTIData"]
        title = obj.get("NIFTIHeader", {}).get("Description", "JNIfTI volume")
        data = container["_ArrayZipData_"]
        dtype_name = container["_ArrayType_"]
        shape = container["_ArraySize_"]
        zip_type = container.get("_ArrayZipType_", "zlib")
        layout = "x_fast"
    elif "Shapes" in obj:
        container = obj["Shapes"]
        title = "Digimouse atlas labels"
        data = container["_ArrayZipData_"]
        dtype_name = container["_ArrayType_"]
        shape = container["_ArraySize_"]
        zip_type = container.get("_ArrayZipType_", "zlib")
        layout = "z_fast"
    else:
        # digimouse.json contains literal newlines inside _ArrayZipData_, so
        # strict JSON parsers reject it. These fields are enough for viewing.
        title = "Digimouse atlas labels"
        data = _regex_value(text, "_ArrayZipData_")
        dtype_name = _regex_value(text, "_ArrayType_")
        shape = _regex_array(text, "_ArraySize_")
        zip_type = _regex_value(text, "_ArrayZipType_")
        layout = "z_fast"

    dtype_name = dtype_name.lower()
    if dtype_name not in DTYPE_INFO:
        raise ValueError(f"Unsupported array type {dtype_name!r}")
    if zip_type.lower() != "zlib":
        raise ValueError(f"Unsupported compression {zip_type!r}")

    encoded = re.sub(r"\s+", "", data)
    raw = zlib.decompress(base64.b64decode(encoded))
    code, itemsize = DTYPE_INFO[dtype_name]
    expected = math.prod(shape) * itemsize
    if len(raw) != expected:
        raise ValueError(f"Decoded {len(raw)} bytes, expected {expected}")

    if code == "B":
        values = raw
    else:
        values = array.array(code)
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()

    return {
        "path": path,
        "title": title,
        "dtype": dtype_name,
        "shape": shape[:3],
        "layout": layout,
        "values": values,
    }


class SliceViewer(tk.Tk):
    def __init__(self, volume):
        super().__init__()
        self.volume = volume
        self.axis = tk.StringVar(value="z")
        self.index = tk.IntVar(value=max(0, volume["shape"][2] // 2))
        self.autoscale = tk.BooleanVar(value=True)
        self.log_scale = tk.BooleanVar(value=volume["dtype"] not in ("uint8", "uchar", "byte"))
        self.photo = None

        self.title(f"{volume['path'].name} - {volume['title']}")
        self.geometry("900x720")
        self._build_ui()
        self._configure_slider()
        self._render()

    def _build_ui(self):
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill="x")

        ttk.Label(controls, text=self.volume["path"].name).pack(side="left")
        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=8)

        for label, axis in (("X", "x"), ("Y", "y"), ("Z", "z")):
            ttk.Radiobutton(
                controls,
                text=label,
                value=axis,
                variable=self.axis,
                command=self._axis_changed,
            ).pack(side="left")

        ttk.Checkbutton(
            controls,
            text="Auto contrast",
            variable=self.autoscale,
            command=self._render,
        ).pack(side="left", padx=12)

        ttk.Checkbutton(
            controls,
            text="Log",
            variable=self.log_scale,
            command=self._render,
        ).pack(side="left")

        self.info = ttk.Label(controls)
        self.info.pack(side="right")

        self.axes_info = ttk.Label(main)
        self.axes_info.pack(anchor="w", pady=(8, 0))

        self.slider = ttk.Scale(main, orient="horizontal", command=self._slider_changed)
        self.slider.pack(fill="x", pady=(8, 4))

        self.canvas = tk.Canvas(main, background="#151515", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render())

    def _axis_changed(self):
        self._configure_slider()
        self._render()

    def _configure_slider(self):
        axis_size = self._axis_size()
        self.index.set(axis_size // 2)
        self.slider.configure(from_=0, to=max(0, axis_size - 1))
        self.slider.set(self.index.get())

    def _slider_changed(self, value):
        self.index.set(int(float(value)))
        self._render()

    def _axis_size(self):
        return self.volume["shape"]["xyz".index(self.axis.get())]

    def _flat_index(self, x, y, z):
        sx, sy, sz = self.volume["shape"]
        if self.volume["layout"] == "x_fast":
            return x + sx * (y + sy * z)
        return (x * sy + y) * sz + z

    def _slice_values(self):
        sx, sy, sz = self.volume["shape"]
        axis = self.axis.get()
        idx = self.index.get()
        vals = self.volume["values"]

        if axis == "x":
            width, height = sy, sz
            labels = ("Y", "Z")
            items = [vals[self._flat_index(idx, y, z)] for z in range(sz) for y in range(sy)]
        elif axis == "y":
            width, height = sx, sz
            labels = ("X", "Z")
            items = [vals[self._flat_index(x, idx, z)] for z in range(sz) for x in range(sx)]
        else:
            width, height = sx, sy
            labels = ("X", "Y")
            items = [vals[self._flat_index(x, y, idx)] for y in range(sy) for x in range(sx)]

        return width, height, labels, items

    def _to_grayscale(self, items):
        if not items:
            return b""

        if self.volume["dtype"] in ("uint8", "uchar", "byte"):
            max_value = max(items) or 1
            return bytes(min(255, int(value * 255 / max_value)) for value in items)

        finite = [value for value in items if math.isfinite(value)]
        if not finite:
            return bytes(len(items))

        if self.log_scale.get():
            positive = [value for value in finite if value > 0.0]
            if not positive:
                return bytes(len(items))
            floor = min(positive)
            finite = [math.log10(max(value, floor)) for value in finite]

        vals = sorted(finite)
        if self.autoscale.get():
            lo = vals[int(0.01 * (len(vals) - 1))]
            hi = vals[int(0.995 * (len(vals) - 1))]
        else:
            lo, hi = vals[0], vals[-1]
        if hi <= lo:
            hi = lo + 1.0
        scale = 255.0 / (hi - lo)

        out = []
        floor = 10 ** lo if self.log_scale.get() else 0.0
        for value in items:
            if not math.isfinite(value):
                out.append(0)
                continue
            if self.log_scale.get():
                value = math.log10(max(value, floor))
            out.append(max(0, min(255, int((value - lo) * scale))))
        return bytes(out)

    def _photo_from_pixels(self, width, height, pixels):
        image = tk.PhotoImage(width=width, height=height)
        rows = []
        for y in range(height):
            start = y * width
            row = pixels[start : start + width]
            colors = " ".join(f"#{value:02x}{value:02x}{value:02x}" for value in row)
            rows.append("{" + colors + "}")
        image.put(" ".join(rows), to=(0, 0, width, height))
        return image

    def _render(self):
        width, height, labels, items = self._slice_values()
        pixels = self._to_grayscale(items)
        self.photo = self._photo_from_pixels(width, height, pixels)

        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = max(1, min(cw // width, ch // height))
        image = self.photo
        if scale > 1:
            image = self.photo.zoom(scale, scale)
            self.photo = image
        self.canvas.create_image(cw // 2, ch // 2, image=image, anchor="center")

        nonzero = sum(1 for value in items if value != 0)
        layout = "x-fast" if self.volume["layout"] == "x_fast" else "z-fast"
        self.axes_info.configure(
            text=f"Horizontal: {labels[0]}   Vertical: {labels[1]}   Storage: {layout}"
        )
        self.info.configure(
            text=(
                f"{self.axis.get().upper()} {self.index.get() + 1}/{self._axis_size()}  "
                f"{width}x{height}  {'log ' if self.log_scale.get() else ''}"
                f"min {min(items):.6g}  max {max(items):.6g}  "
                f"nonzero {nonzero}"
            )
        )


def main():
    default_file = "digimouse.jnii" if Path("digimouse.jnii").exists() else "digimouse.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file",
        nargs="?",
        default=default_file,
        help="digimouse.json or digimouse.jnii",
    )
    args = parser.parse_args()

    volume = load_volume(Path(args.file))
    SliceViewer(volume).mainloop()


if __name__ == "__main__":
    main()
