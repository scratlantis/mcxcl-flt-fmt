#!/usr/bin/env python3
"""Tk slice viewer for Digimouse Jacobian/truth NumPy volumes."""

from __future__ import annotations

import argparse
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np


class NumpySliceViewer(tk.Tk):
    def __init__(
        self,
        path: Path,
        data: np.ndarray,
        weight_path: Path | None = None,
        weight_data: np.ndarray | None = None,
        export_dir: Path | None = None,
        reference_path: Path | None = None,
        reference_data: np.ndarray | None = None,
    ):
        super().__init__()
        self.path = path
        self.data = np.asarray(data)
        self.weight_path = weight_path
        self.weight_data = None if weight_data is None else np.asarray(weight_data)
        self.reference_path = reference_path
        self.reference_data = None if reference_data is None else np.asarray(reference_data)
        self.export_dir = export_dir or path.parent / "marked_source"
        self.array = self._initial_volume()
        self.reference_array = self._initial_reference_volume()
        self.view_mode = tk.StringVar(value="marked")
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
        self.export_text = tk.StringVar()
        self.value_log_bounds = self._log_bounds()
        self.photo = None

        self.title(f"{path.name} - NumPy slice viewer")
        self.geometry("940x830")
        self._build_ui()
        self._configure_selectors()
        self._configure_slider()
        self._configure_mark_sliders(reset=True)
        self._configure_comparison()
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

    def _initial_reference_volume(self) -> np.ndarray | None:
        if self.reference_data is None:
            return None
        if self.reference_data.ndim == 3:
            volume = self.reference_data
        elif self.reference_data.ndim == 4:
            volume = self.reference_data[0]
        elif self.reference_data.ndim == 5:
            volume = self.reference_data[0, 0]
        else:
            raise ValueError(
                f"expected a 3D, 4D, or 5D reference array, got {self.reference_data.shape}"
            )
        if volume.shape != self.array.shape:
            raise ValueError(f"reference shape {volume.shape} does not match data shape {self.array.shape}")
        return volume

    def _select_reference_volume(self) -> np.ndarray | None:
        if self.reference_data is None:
            return None
        if self.reference_data.ndim == 3:
            volume = self.reference_data
        elif self.reference_data.ndim == 4:
            volume = self.reference_data[self.detector_index.get()]
        else:
            volume = self.reference_data[self.source_index.get(), self.detector_index.get()]
        if volume.shape != self.array.shape:
            raise ValueError(f"reference shape {volume.shape} does not match data shape {self.array.shape}")
        return volume

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

        self.comparison_controls = ttk.Frame(main)
        self.comparison_controls.pack(fill="x", pady=(8, 0))
        ttk.Label(self.comparison_controls, text="View").pack(side="left")
        for label, mode in (("Original", "original"), ("Marked source", "marked"), ("Error", "error")):
            ttk.Radiobutton(
                self.comparison_controls,
                text=label,
                value=mode,
                variable=self.view_mode,
                command=self._comparison_changed,
            ).pack(side="left", padx=(8, 0))
        self.error_info = ttk.Label(self.comparison_controls, text="Error = marked source - original")
        self.error_info.pack(side="right")

        self.slider = ttk.Scale(main, orient="horizontal", command=self._slider_changed)
        self.slider.pack(fill="x", pady=(8, 4))

        self.marks_frame = ttk.Frame(main)
        self.marks_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(self.marks_frame, textvariable=self.s1_text, width=17).grid(row=0, column=0, sticky="w")
        self.s1_slider = ttk.Scale(
            self.marks_frame, orient="horizontal", variable=self.s1_exponent, command=self._mark_changed
        )
        self.s1_slider.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(self.marks_frame, textvariable=self.s2_text, width=17).grid(row=1, column=0, sticky="w")
        self.s2_slider = ttk.Scale(
            self.marks_frame, orient="horizontal", variable=self.s2_exponent, command=self._mark_changed
        )
        self.s2_slider.grid(row=1, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(self.marks_frame, textvariable=self.marked_text, width=22, anchor="e").grid(
            row=0, column=2, rowspan=2, sticky="e"
        )
        self.export_button = ttk.Button(
            self.marks_frame, text="Export marked source", command=self._export_marked_source
        )
        self.export_button.grid(row=0, column=3, rowspan=2, padx=(8, 0))
        self.marks_frame.columnconfigure(1, weight=1)

        self.export_label = ttk.Label(main, textvariable=self.export_text)
        self.export_label.pack(fill="x", pady=(0, 4))

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

    def _configure_comparison(self) -> None:
        if self.reference_array is None:
            self.comparison_controls.pack_forget()
            return
        self.log_scale.set(False)
        self.marks_frame.pack_forget()
        self.export_label.pack_forget()

    def _comparison_changed(self) -> None:
        self._render()

    def _axis_changed(self) -> None:
        self._configure_slider()
        self._render()

    def _volume_changed(self) -> None:
        self.array = self._select_volume()
        self.reference_array = self._select_reference_volume()
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

    def _select_weight_volume(self) -> np.ndarray:
        if self.weight_data is None:
            raise ValueError("no weight volume was loaded")
        if self.weight_data.ndim == 3:
            weights = self.weight_data
        elif self.weight_data.ndim == 4:
            weights = self.weight_data[self.detector_index.get()]
        elif self.weight_data.ndim == 5:
            weights = self.weight_data[self.source_index.get(), self.detector_index.get()]
        else:
            raise ValueError(f"expected a 3D, 4D, or 5D weight array, got {self.weight_data.shape}")
        if weights.shape != self.array.shape:
            raise ValueError(f"weight shape {weights.shape} does not match value shape {self.array.shape}")
        return np.asarray(weights)

    def _export_marked_source(self) -> None:
        try:
            weights = self._select_weight_volume()
            values = np.asarray(self.array)
            s1, s2 = self._mark_values()
            mask = np.isfinite(values) & (np.abs(values - s1) < s2)
            valid_weights = np.isfinite(weights) & (weights > 0.0)
            source = np.where(mask & valid_weights, weights, 0.0).astype(np.float32)
            weighted_count = int(np.count_nonzero(source))
            total_strength = float(np.sum(source, dtype=np.float64))
            if weighted_count == 0 or not math.isfinite(total_strength) or total_strength <= 0.0:
                raise ValueError("the current marked region contains no voxels with positive finite fluence")

            self.export_dir.mkdir(parents=True, exist_ok=True)
            mask_path = self.export_dir / "marked_mask.npy"
            source_path = self.export_dir / "source_weights.npy"
            binary_path = self.export_dir / "source_weights.bin"
            metadata_path = self.export_dir / "marked_source.json"
            np.save(mask_path, mask)
            np.save(source_path, source)
            source.ravel(order="F").tofile(binary_path)
            metadata = {
                "value_volume": str(self.path.resolve()),
                "weight_volume": str(self.weight_path.resolve()) if self.weight_path else None,
                "shape": list(source.shape),
                "source_index": self.source_index.get(),
                "detector_index": self.detector_index.get(),
                "s1": s1,
                "s2": s2,
                "marked_voxels": int(np.count_nonzero(mask)),
                "weighted_voxels": weighted_count,
                "total_relative_source_strength": total_strength,
                "mask": str(mask_path.resolve()),
                "source_weights": str(source_path.resolve()),
                "source_binary": str(binary_path.resolve()),
                "binary_order": "Fortran/x-fast",
                "binary_dtype": "float32 little-endian",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            self.export_text.set(
                f"Exported {weighted_count:,} weighted voxels, total strength {total_strength:.6g}"
            )
        except Exception as exc:
            messagebox.showerror("Could not export marked source", str(exc), parent=self)

    def _update_mark_labels(self) -> None:
        s1, s2 = self._mark_values()
        self.s1_text.set(f"s1  {s1:.6g}")
        self.s2_text.set(f"s2  {s2:.6g}")

    def _slice_volume(self, volume: np.ndarray) -> tuple[np.ndarray, tuple[str, str]]:
        idx = self.index.get()
        if self.axis.get() == "x":
            return volume[idx, :, :].T, ("Y", "Z")
        if self.axis.get() == "y":
            return volume[:, idx, :].T, ("X", "Z")
        return volume[:, :, idx].T, ("X", "Y")

    def _slice(self) -> tuple[np.ndarray, tuple[str, str]]:
        if self.reference_array is None or self.view_mode.get() == "marked":
            return self._slice_volume(self.array)
        reference, labels = self._slice_volume(self.reference_array)
        if self.view_mode.get() == "original":
            return reference, labels
        marked, _ = self._slice_volume(self.array)
        return marked.astype(np.float64) - reference.astype(np.float64), labels

    def _error_to_rgb(self, image: np.ndarray) -> bytes:
        values = image.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        if not np.any(finite):
            return bytes(values.size * 3)
        magnitudes = np.abs(values[finite])
        limit = float(np.percentile(magnitudes, 99.5)) if self.autoscale.get() else float(np.max(magnitudes))
        if limit <= 0.0:
            limit = 1.0
        scaled = np.clip(values / limit, -1.0, 1.0)
        strength = np.abs(scaled)
        rgb = np.empty(values.shape + (3,), dtype=np.float64)
        base = 24.0 + 55.0 * (1.0 - strength)
        rgb[..., 0] = base
        rgb[..., 1] = base
        rgb[..., 2] = base
        positive = scaled > 0.0
        negative = scaled < 0.0
        rgb[..., 0][positive] = 55.0 + 200.0 * strength[positive]
        rgb[..., 1][positive] = 35.0 * (1.0 - strength[positive])
        rgb[..., 2][positive] = 35.0 * (1.0 - strength[positive])
        rgb[..., 0][negative] = 35.0 * (1.0 - strength[negative])
        rgb[..., 1][negative] = 65.0 + 75.0 * strength[negative]
        rgb[..., 2][negative] = 70.0 + 185.0 * strength[negative]
        rgb[~finite] = 0.0
        return np.clip(rgb, 0, 255).astype(np.uint8).tobytes(order="C")

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

        marked = np.zeros(values.shape, dtype=bool)
        if self.reference_array is None:
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
        if self.reference_array is not None and self.view_mode.get() == "error":
            pixels = self._error_to_rgb(image)
            marked_count = 0
        else:
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
        if self.reference_array is not None:
            self.marked_text.set(self.view_mode.get().replace("_", " ").title())
        else:
            self.marked_text.set(f"Marked: {marked_count:,} / {image.size:,}")
        self.axes_info.configure(text=f"Horizontal: {labels[0]}   Vertical: {labels[1]}   Shape: {self.data.shape}")
        self.info.configure(
            text=(
                f"{self.view_mode.get().replace('_', ' ').title()}  "
                if self.reference_array is not None
                else ""
            )
            + (
                f"{self.axis.get().upper()} {self.index.get() + 1}/{self.array.shape['xyz'.index(self.axis.get())]}  "
                f"min {min_value:.6g}  max {max_value:.6g}"
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="3D, 4D, or 5D .npy volume")
    parser.add_argument(
        "--weight-volume",
        type=Path,
        help="fluence volume used to weight marked voxels (defaults to sibling fluence.npy)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="directory for marked mask and source weights (defaults to sibling marked_source)",
    )
    parser.add_argument(
        "--reference-volume",
        type=Path,
        help="original volume used for original/marked/error comparison modes",
    )
    args = parser.parse_args()
    path = args.file.resolve()
    weight_path = args.weight_volume
    if weight_path is None:
        candidate = path.parent / "fluence.npy"
        weight_path = candidate if candidate.exists() else None
    elif not weight_path.is_absolute():
        weight_path = weight_path.resolve()
    export_dir = args.export_dir.resolve() if args.export_dir else None
    reference_path = args.reference_volume.resolve() if args.reference_volume else None
    weight_data = np.load(weight_path, mmap_mode="r") if weight_path else None
    viewer = NumpySliceViewer(
        path,
        np.load(path, mmap_mode="r"),
        weight_path=weight_path,
        weight_data=weight_data,
        export_dir=export_dir,
        reference_path=reference_path,
        reference_data=np.load(reference_path, mmap_mode="r") if reference_path else None,
    )
    if weight_path is None:
        viewer.export_button.configure(state="disabled")
        viewer.export_text.set("Load --weight-volume to export a fluence-weighted source")
    else:
        viewer.export_text.set(f"Source weights: {weight_path.name}")
    viewer.mainloop()


if __name__ == "__main__":
    main()
