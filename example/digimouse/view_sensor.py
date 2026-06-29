#!/usr/bin/env python3
"""Tk viewer for adjoint sensor measurements.

Left pane  — 3D slice of  phi_fwd[v] * phi_adj_d[v]  at marked voxels.
Right pane — 2D heatmap of the summed measurement per detector.

Usage:
    python view_sensor.py [adjoint_sensor_output/]

Clicking a cell in the heatmap or using the row/col spinboxes selects
the active detector and updates the slice view accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np


class SensorViewer(tk.Tk):
    def __init__(
        self,
        output_dir: Path,
        masked_forward: np.ndarray,
        masked_adjoint: np.ndarray,
        marked_mask: np.ndarray,
        det_positions: np.ndarray,
        rows: int,
        cols: int,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.masked_forward = masked_forward.astype(np.float64)
        self.masked_adjoint = masked_adjoint.astype(np.float64)
        self.marked_mask = marked_mask.astype(bool)
        self.det_positions = det_positions  # [rows, cols, 3]
        self.shape = marked_mask.shape
        self.rows = rows
        self.cols = cols

        # Precompute all detector measurements: [n_dets]
        self.measurements = self.masked_adjoint @ self.masked_forward
        self.meas_grid = self.measurements.reshape(rows, cols)

        # State
        self.det_row = tk.IntVar(value=0)
        self.det_col = tk.IntVar(value=0)
        self.axis = tk.StringVar(value="z")
        self.index = tk.IntVar(value=self.shape[2] // 2)
        self.autoscale = tk.BooleanVar(value=True)
        self.log_scale = tk.BooleanVar(value=True)
        self.heatmap_log = tk.BooleanVar(value=True)

        self.slice_photo: tk.PhotoImage | None = None
        self.heat_photo: tk.PhotoImage | None = None

        self.title(f"{output_dir.name} — adjoint sensor viewer")
        self.geometry("1300x780")
        self._build_ui()
        self.slice_canvas.bind("<Configure>", lambda _e: self._render_slice())
        self.heat_canvas.bind("<Configure>", lambda _e: self._render_heatmap())
        self.after(10, self._render_all)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _det_idx(self) -> int:
        return self.det_row.get() * self.cols + self.det_col.get()

    def _product_volume(self) -> np.ndarray:
        idx = self._det_idx()
        vol = np.zeros(self.shape, dtype=np.float64)
        vol[self.marked_mask] = self.masked_forward * self.masked_adjoint[idx]
        return vol

    def _slice_volume(self, vol: np.ndarray) -> tuple[np.ndarray, tuple[str, str]]:
        i = self.index.get()
        if self.axis.get() == "x":
            return vol[i, :, :].T, ("Y", "Z")
        if self.axis.get() == "y":
            return vol[:, i, :].T, ("X", "Z")
        return vol[:, :, i].T, ("X", "Y")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=6)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x")
        ttk.Label(top, text=self.output_dir.name).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        for label, axis in (("X", "x"), ("Y", "y"), ("Z", "z")):
            ttk.Radiobutton(
                top, text=label, value=axis, variable=self.axis, command=self._axis_changed
            ).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Checkbutton(
            top, text="Auto contrast", variable=self.autoscale, command=self._render_slice
        ).pack(side="left")
        ttk.Checkbutton(
            top, text="Log (slice)", variable=self.log_scale, command=self._render_slice
        ).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(
            top, text="Log (grid)", variable=self.heatmap_log, command=self._render_heatmap
        ).pack(side="left", padx=(6, 0))
        self.info_label = ttk.Label(top)
        self.info_label.pack(side="right")

        panes = ttk.PanedWindow(main, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(6, 0))

        self.slice_canvas = tk.Canvas(panes, background="#151515", highlightthickness=0)
        panes.add(self.slice_canvas, weight=3)

        right = ttk.Frame(panes, padding=(6, 0, 0, 0))
        panes.add(right, weight=1)
        ttk.Label(right, text="Sensor measurements").pack(anchor="w", pady=(0, 2))
        self.heat_canvas = tk.Canvas(
            right,
            background="#151515",
            highlightthickness=0,
            width=max(200, self.cols * 24),
            height=max(200, self.rows * 24),
        )
        self.heat_canvas.pack(fill="both", expand=True)
        self.heat_canvas.bind("<Button-1>", self._heatmap_click)
        self.det_meas_label = ttk.Label(right)
        self.det_meas_label.pack(anchor="w", pady=(4, 0))

        bot = ttk.Frame(main)
        bot.pack(fill="x", pady=(4, 0))
        ttk.Label(bot, text="Slice:").pack(side="left")
        self.slice_slider = ttk.Scale(bot, orient="horizontal", command=self._slice_changed)
        self.slice_slider.pack(side="left", fill="x", expand=True, padx=(4, 12))
        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(bot, text="Detector  row").pack(side="left")
        ttk.Spinbox(
            bot,
            from_=0,
            to=self.rows - 1,
            width=4,
            textvariable=self.det_row,
            command=self._detector_changed,
        ).pack(side="left", padx=(4, 8))
        ttk.Label(bot, text="col").pack(side="left")
        ttk.Spinbox(
            bot,
            from_=0,
            to=self.cols - 1,
            width=4,
            textvariable=self.det_col,
            command=self._detector_changed,
        ).pack(side="left", padx=(4, 0))

        self._configure_slider()

    def _configure_slider(self) -> None:
        axis_size = self.shape["xyz".index(self.axis.get())]
        self.index.set(min(self.index.get(), axis_size - 1))
        self.slice_slider.configure(from_=0, to=max(0, axis_size - 1))
        self.slice_slider.set(self.index.get())

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def _axis_changed(self) -> None:
        self._configure_slider()
        self._render_slice()

    def _slice_changed(self, value: str) -> None:
        self.index.set(int(float(value)))
        self._render_slice()

    def _detector_changed(self) -> None:
        self._render_all()

    def _heatmap_click(self, event: tk.Event) -> None:
        cw = max(1, self.heat_canvas.winfo_width())
        ch = max(1, self.heat_canvas.winfo_height())
        sx = max(1, cw // self.cols)
        sy = max(1, ch // self.rows)
        scale = min(sx, sy)
        img_w = self.cols * scale
        img_h = self.rows * scale
        left = (cw - img_w) // 2
        top = (ch - img_h) // 2
        col = (event.x - left) // scale
        row = (event.y - top) // scale
        col = max(0, min(self.cols - 1, int(col)))
        row = max(0, min(self.rows - 1, int(row)))
        self.det_col.set(col)
        self.det_row.set(row)
        self._render_all()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _to_gray_rgb(self, image: np.ndarray) -> bytes:
        values = image.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        if not np.any(finite):
            return bytes(values.size * 3)
        shown = values.copy()
        if self.log_scale.get():
            positive = finite & (shown > 0)
            if not np.any(positive):
                return bytes(values.size * 3)
            floor = float(np.min(shown[positive]))
            shown = np.log10(np.maximum(shown, floor))
            finite = np.isfinite(shown)
        vals = shown[finite]
        if self.autoscale.get():
            lo = float(np.percentile(vals, 1.0))
            hi = float(np.percentile(vals, 99.5))
        else:
            lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((shown - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
        scaled[~finite] = 0.0
        gray = scaled.astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2).tobytes(order="C")

    def _to_heat_rgb(self, grid: np.ndarray) -> bytes:
        """Black→red→yellow thermal colormap for the sensor grid."""
        values = grid.astype(np.float64)
        out = np.zeros((*grid.shape, 3), dtype=np.uint8)
        finite = np.isfinite(values) & (values > 0)
        if np.any(finite):
            shown = values.copy()
            if self.heatmap_log.get():
                shown[finite] = np.log10(values[finite])
                shown[~finite] = np.nan
                finite = np.isfinite(shown)
            if np.any(finite):
                lo = float(np.min(shown[finite]))
                hi = float(np.max(shown[finite]))
                if hi <= lo:
                    hi = lo + 1.0
                t = np.zeros_like(shown)
                t[finite] = np.clip((shown[finite] - lo) / (hi - lo), 0.0, 1.0)
                out[..., 0] = np.clip(t * 2.0 * 255, 0, 255).astype(np.uint8)
                out[..., 1] = np.clip((t * 2.0 - 1.0) * 255, 0, 255).astype(np.uint8)
                out[..., 2] = 0
        return out.tobytes(order="C")

    @staticmethod
    def _photo_from_pixels(width: int, height: int, pixels: bytes) -> tk.PhotoImage:
        image = tk.PhotoImage(width=width, height=height)
        rows_data = []
        for y in range(height):
            row = pixels[y * width * 3 : (y + 1) * width * 3]
            rows_data.append(
                "{"
                + " ".join(
                    f"#{row[off]:02x}{row[off + 1]:02x}{row[off + 2]:02x}"
                    for off in range(0, len(row), 3)
                )
                + "}"
            )
        image.put(" ".join(rows_data), to=(0, 0, width, height))
        return image

    def _render_slice(self) -> None:
        vol = self._product_volume()
        image, ax_labels = self._slice_volume(vol)
        h, w = image.shape
        if w == 0 or h == 0:
            return

        pixels = self._to_gray_rgb(image)
        photo = self._photo_from_pixels(w, h, pixels)
        cw = max(1, self.slice_canvas.winfo_width())
        ch = max(1, self.slice_canvas.winfo_height())
        scale = max(1, min(cw // w, ch // h))
        if scale > 1:
            photo = photo.zoom(scale, scale)
        self.slice_photo = photo
        self.slice_canvas.delete("all")
        self.slice_canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")

        positive = image[np.isfinite(image) & (image > 0)]
        max_val = float(np.max(positive)) if positive.size else math.nan
        det_idx = self._det_idx()
        meas = float(self.measurements[det_idx])
        axis = self.axis.get()
        ax_size = self.shape["xyz".index(axis)]
        self.info_label.configure(
            text=f"det ({self.det_row.get()},{self.det_col.get()})  "
                 f"{axis.upper()} {self.index.get() + 1}/{ax_size}  "
                 f"slice max {max_val:.4g}  M {meas:.4g}"
        )

    def _render_heatmap(self) -> None:
        pixels = self._to_heat_rgb(self.meas_grid)
        photo = self._photo_from_pixels(self.cols, self.rows, pixels)

        cw = max(1, self.heat_canvas.winfo_width())
        ch = max(1, self.heat_canvas.winfo_height())
        sx = max(1, cw // self.cols)
        sy = max(1, ch // self.rows)
        scale = min(sx, sy)
        if scale > 1:
            photo = photo.zoom(scale, scale)
        self.heat_photo = photo
        self.heat_canvas.delete("all")

        img_w = self.cols * scale
        img_h = self.rows * scale
        left = (cw - img_w) // 2
        top = (ch - img_h) // 2
        self.heat_canvas.create_image(left, top, image=photo, anchor="nw")

        sel_row = self.det_row.get()
        sel_col = self.det_col.get()
        x0 = left + sel_col * scale
        y0 = top + sel_row * scale
        self.heat_canvas.create_rectangle(
            x0, y0, x0 + scale, y0 + scale, outline="#00c8ff", width=2
        )

        det_idx = self._det_idx()
        pos = self.det_positions[sel_row, sel_col]
        meas = float(self.measurements[det_idx])
        self.det_meas_label.configure(
            text=f"pos [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]  M = {meas:.4g}"
        )

    def _render_all(self) -> None:
        self._render_slice()
        self._render_heatmap()


# ------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=here / "adjoint_sensor_output",
        help="directory containing adjoint_sensor_run.json and .npy outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    meta_path = output_dir / "adjoint_sensor_run.json"
    if not meta_path.exists():
        raise SystemExit(f"not found: {meta_path}\nRun run_adjoint_sensor.py first.")
    meta = json.loads(meta_path.read_text())

    masked_forward = np.load(output_dir / "masked_forward.npy")
    masked_adjoint = np.load(output_dir / "masked_adjoint.npy")
    marked_mask = np.load(output_dir / "marked_mask.npy").astype(bool)
    det_positions = np.load(output_dir / "detector_positions.npy")

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    n_dets = rows * cols

    if masked_adjoint.shape != (n_dets, masked_forward.shape[0]):
        raise ValueError(
            f"masked_adjoint shape {masked_adjoint.shape} does not match "
            f"expected ({n_dets}, {masked_forward.shape[0]})"
        )

    viewer = SensorViewer(output_dir, masked_forward, masked_adjoint, marked_mask, det_positions, rows, cols)
    viewer.mainloop()


if __name__ == "__main__":
    main()
