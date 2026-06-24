#!/usr/bin/env python3
"""Tk slice viewer for Digimouse Jacobian/truth NumPy volumes."""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np


class NumpySliceViewer(tk.Tk):
    def __init__(self, path: Path, data: np.ndarray):
        super().__init__()
        self.path = path
        self.data = np.asarray(data)
        self.array = self._initial_volume()
        self.axis = tk.StringVar(value="z")
        self.index = tk.IntVar(value=max(0, self.array.shape[2] // 2))
        self.source_index = tk.IntVar(value=0)
        self.detector_index = tk.IntVar(value=0)
        self.autoscale = tk.BooleanVar(value=True)
        default_log = any(token in path.stem.lower() for token in ("ratio", "moment", "fluence", "effective"))
        self.log_scale = tk.BooleanVar(value=default_log)
        self.s1_exponent = tk.DoubleVar(value=0.0)
        self.s2_exponent = tk.DoubleVar(value=-1.0)
        self.s1_text = tk.StringVar()
        self.s2_text = tk.StringVar()
        self.marked_text = tk.StringVar()
        self.value_log_bounds = self._log_bounds()
        self.photo = None

        self.title(f"{path.name} - NumPy slice viewer")
        self.geometry("940x830")
        self._build_ui()
        self._configure_selectors()
        self._configure_slider()
        self._configure_mark_sliders(reset=True)
        self._render()

    def _initial_volume(self) -> np.ndarray:
        if self.data.ndim == 3:
            return self.data
        if self.data.ndim == 4:
            return self.data[0]
        if self.data.ndim == 5:
            return self.data[0, 0]
        raise ValueError(f"expected a 3D, 4D, or 5D array, got {self.data.shape}")

    def _select_volume(self) -> np.ndarray:
        if self.data.ndim == 3:
            return self.data
        if self.data.ndim == 4:
            return self.data[self.detector_index.get()]
        return self.data[self.source_index.get(), self.detector_index.get()]

    def _log_bounds(self) -> tuple[float, float]:
        values = np.asarray(self.array)
        positive = values[np.isfinite(values) & (values > 0)]
        if not positive.size:
            return -12.0, 0.0
        lo = math.floor(math.log10(float(np.min(positive))))
        hi = math.ceil(math.log10(float(np.max(positive))))
        return float(lo), float(max(lo + 1.0, hi))

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill="x")
        ttk.Label(controls, text=self.path.name).pack(side="left")
        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=8)

        for label, axis in (("X", "x"), ("Y", "y"), ("Z", "z")):
            ttk.Radiobutton(
                controls,
                text=label,
                value=axis,
                variable=self.axis,
                command=self._axis_changed,
            ).pack(side="left")

        ttk.Checkbutton(controls, text="Auto contrast", variable=self.autoscale, command=self._render).pack(
            side="left", padx=12
        )
        ttk.Checkbutton(controls, text="Log", variable=self.log_scale, command=self._render).pack(side="left")
        self.info = ttk.Label(controls)
        self.info.pack(side="right")

        selectors = ttk.Frame(main)
        selectors.pack(fill="x", pady=(8, 0))
        ttk.Label(selectors, text="Source").pack(side="left")
        self.source_spin = ttk.Spinbox(
            selectors,
            from_=0,
            to=0,
            width=5,
            textvariable=self.source_index,
            command=self._volume_changed,
        )
        self.source_spin.pack(side="left", padx=(4, 14))
        ttk.Label(selectors, text="Detector").pack(side="left")
        self.detector_spin = ttk.Spinbox(
            selectors,
            from_=0,
            to=0,
            width=5,
            textvariable=self.detector_index,
            command=self._volume_changed,
        )
        self.detector_spin.pack(side="left", padx=(4, 14))
        self.axes_info = ttk.Label(selectors)
        self.axes_info.pack(side="right")

        self.slider = ttk.Scale(main, orient="horizontal", command=self._slider_changed)
        self.slider.pack(fill="x", pady=(8, 4))

        marks = ttk.Frame(main)
        marks.pack(fill="x", pady=(4, 8))
        ttk.Label(marks, textvariable=self.s1_text, width=17).grid(row=0, column=0, sticky="w")
        self.s1_slider = ttk.Scale(marks, orient="horizontal", variable=self.s1_exponent, command=self._mark_changed)
        self.s1_slider.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(marks, textvariable=self.s2_text, width=17).grid(row=1, column=0, sticky="w")
        self.s2_slider = ttk.Scale(marks, orient="horizontal", variable=self.s2_exponent, command=self._mark_changed)
        self.s2_slider.grid(row=1, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(marks, textvariable=self.marked_text, width=22, anchor="e").grid(
            row=0, column=2, rowspan=2, sticky="e"
        )
        marks.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(main, background="#151515", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render())

    def _configure_selectors(self) -> None:
        if self.data.ndim == 5:
            self.source_spin.configure(from_=0, to=self.data.shape[0] - 1, state="normal")
            self.detector_spin.configure(from_=0, to=self.data.shape[1] - 1, state="normal")
        elif self.data.ndim == 4:
            self.source_spin.configure(from_=0, to=0, state="disabled")
            self.detector_spin.configure(from_=0, to=self.data.shape[0] - 1, state="normal")
        else:
            self.source_spin.configure(from_=0, to=0, state="disabled")
            self.detector_spin.configure(from_=0, to=0, state="disabled")

    def _axis_changed(self) -> None:
        self._configure_slider()
        self._render()

    def _volume_changed(self) -> None:
        self.array = self._select_volume()
        self.value_log_bounds = self._log_bounds()
        self._configure_slider()
        self._configure_mark_sliders(reset=True)
        self._render()

    def _configure_slider(self) -> None:
        axis_size = self.array.shape["xyz".index(self.axis.get())]
        self.index.set(min(self.index.get(), axis_size - 1))
        self.slider.configure(from_=0, to=max(0, axis_size - 1))
        self.slider.set(self.index.get())

    def _slider_changed(self, value: str) -> None:
        self.index.set(int(float(value)))
        self._render()

    def _configure_mark_sliders(self, reset: bool) -> None:
        lo, hi = self.value_log_bounds
        slider_lo = lo - 2.0
        slider_hi = hi + 1.0
        self.s1_slider.configure(from_=slider_lo, to=slider_hi)
        self.s2_slider.configure(from_=slider_lo, to=slider_hi)
        if reset:
            self.s1_exponent.set((lo + hi) / 2.0)
            self.s2_exponent.set(max(slider_lo, self.s1_exponent.get() - 1.0))
        self._update_mark_labels()

    def _mark_changed(self, _value: str = "") -> None:
        self._update_mark_labels()
        self._render()

    def _mark_values(self) -> tuple[float, float]:
        return 10.0 ** self.s1_exponent.get(), 10.0 ** self.s2_exponent.get()

    def _update_mark_labels(self) -> None:
        s1, s2 = self._mark_values()
        self.s1_text.set(f"s1  {s1:.6g}")
        self.s2_text.set(f"s2  {s2:.6g}")

    def _slice(self) -> tuple[np.ndarray, tuple[str, str]]:
        idx = self.index.get()
        if self.axis.get() == "x":
            return self.array[idx, :, :].T, ("Y", "Z")
        if self.axis.get() == "y":
            return self.array[:, idx, :].T, ("X", "Z")
        return self.array[:, :, idx].T, ("X", "Y")

    def _to_rgb(self, image: np.ndarray) -> tuple[bytes, int]:
        values = image.astype(np.float64, copy=False)
        raw_finite = np.isfinite(values)
        if not np.any(raw_finite):
            return bytes(values.size * 3), 0
        shown = values.copy()
        display_finite = raw_finite.copy()
        if self.log_scale.get():
            positive = raw_finite & (shown > 0)
            if not np.any(positive):
                return bytes(values.size * 3), 0
            floor = float(np.min(shown[positive]))
            shown = np.log10(np.maximum(shown, floor))
            display_finite = np.isfinite(shown)
        vals = shown[display_finite]
        if self.autoscale.get():
            lo = float(np.percentile(vals, 1.0))
            hi = float(np.percentile(vals, 99.5))
        else:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((shown - lo) * (255.0 / (hi - lo)), 0, 255)
        scaled[~display_finite] = 0
        gray = scaled.astype(np.float64)
        rgb = np.repeat(gray[..., None], 3, axis=2)

        s1, s2 = self._mark_values()
        distance = np.abs(values - s1)
        marked = raw_finite & (distance < s2)
        if np.any(marked):
            closeness = np.clip(1.0 - distance[marked] / s2, 0.0, 1.0)
            alpha = 0.3 + 0.7 * closeness
            target = np.array([20.0, 90.0, 255.0])
            rgb[marked] = rgb[marked] * (1.0 - alpha[:, None]) + target * alpha[:, None]

        return np.clip(rgb, 0, 255).astype(np.uint8).tobytes(order="C"), int(np.count_nonzero(marked))

    def _photo_from_pixels(self, width: int, height: int, pixels: bytes) -> tk.PhotoImage:
        image = tk.PhotoImage(width=width, height=height)
        rows = []
        for y in range(height):
            row = pixels[y * width * 3 : (y + 1) * width * 3]
            rows.append(
                "{"
                + " ".join(
                    f"#{row[offset]:02x}{row[offset + 1]:02x}{row[offset + 2]:02x}"
                    for offset in range(0, len(row), 3)
                )
                + "}"
            )
        image.put(" ".join(rows), to=(0, 0, width, height))
        return image

    def _render(self) -> None:
        image, labels = self._slice()
        height, width = image.shape
        pixels, marked_count = self._to_rgb(image)
        self.photo = self._photo_from_pixels(width, height, pixels)

        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = max(1, min(cw // width, ch // height))
        photo = self.photo.zoom(scale, scale) if scale > 1 else self.photo
        self.photo = photo
        self.canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")

        finite = image[np.isfinite(image)]
        min_value = float(np.min(finite)) if finite.size else math.nan
        max_value = float(np.max(finite)) if finite.size else math.nan
        self.marked_text.set(f"Marked: {marked_count:,} / {image.size:,}")
        self.axes_info.configure(text=f"Horizontal: {labels[0]}   Vertical: {labels[1]}   Shape: {self.data.shape}")
        self.info.configure(
            text=(
                f"{self.axis.get().upper()} {self.index.get() + 1}/{self.array.shape['xyz'.index(self.axis.get())]}  "
                f"min {min_value:.6g}  max {max_value:.6g}"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="3D, 4D, or 5D .npy volume")
    args = parser.parse_args()
    NumpySliceViewer(args.file, np.load(args.file, mmap_mode="r")).mainloop()


if __name__ == "__main__":
    main()
