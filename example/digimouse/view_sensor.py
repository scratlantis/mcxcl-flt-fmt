#!/usr/bin/env python3
"""Tk viewer for adjoint sensor measurements.

Left pane  — 3D composite slice: tissue labels (background) + phi_fwd*phi_adj_d
             at marked voxels (warm overlay), source cross and detector circles.
Right pane — 2D thermal heatmap of the summed measurement per detector.

Usage:
    python view_sensor.py [adjoint_sensor_output/]

Clicking a heatmap cell or using the row/col spinboxes selects the active detector.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import tkinter as tk
import zlib
from pathlib import Path
from tkinter import ttk

import numpy as np


# ------------------------------------------------------------------
# label helpers
# ------------------------------------------------------------------

def load_json_relaxed(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    def compact_zip(match: re.Match[str]) -> str:
        compact = re.sub(r"\s+", "", match.group(2))
        return f'"{match.group(1)}": "{compact}"'

    fixed = re.sub(r'"(_ArrayZipData_)"\s*:\s*"([^"]*)"', compact_zip, text, flags=re.S)
    return json.loads(fixed)


def decode_shapes(config: dict) -> np.ndarray:
    shapes = config["Shapes"]
    if shapes.get("_ArrayZipType_", "zlib").lower() != "zlib":
        raise ValueError("only zlib-compressed Shapes arrays are supported")
    shape = tuple(int(v) for v in shapes["_ArraySize_"][:3])
    raw = zlib.decompress(base64.b64decode(re.sub(r"\s+", "", shapes["_ArrayZipData_"])))
    if len(raw) != math.prod(shape):
        raise ValueError(f"decoded {len(raw)} label bytes, expected {math.prod(shape)}")
    # atlas uses z-fast (C order): index = x*Ny*Nz + y*Nz + z
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape, order="C").copy()


def _make_label_colors(n_labels: int) -> np.ndarray:
    """Golden-angle hue LUT for tissue labels; dim saturation for background context."""
    colors = np.zeros((n_labels, 3), dtype=np.uint8)
    colors[0] = [12, 12, 18]  # vacuum / background
    for i in range(1, n_labels):
        h = (i * 137.508) % 360.0
        h6, s, v = h / 60.0, 0.55, 0.48
        c = v * s
        x = c * (1.0 - abs(h6 % 2.0 - 1.0))
        m = v - c
        seg = int(h6)
        pairs = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
        r, g, b = pairs[min(seg, 5)]
        colors[i] = [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]
    return colors


# ------------------------------------------------------------------
# viewer
# ------------------------------------------------------------------

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
        labels: np.ndarray | None = None,
        source_pos: list[float] | None = None,
        path_sens_adj: np.ndarray | None = None,
        mua_orig: np.ndarray | None = None,
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
        self.labels = labels  # [Nx, Ny, Nz] uint8, or None
        self.source_pos = source_pos  # [3] float, or None
        self.label_colors = (
            _make_label_colors(int(labels.max()) + 1) if labels is not None else None
        )
        self.path_sens_adj = path_sens_adj  # [n_dets, n_media] or None
        self.mua_orig = mua_orig  # [n_media] mm⁻¹, or None

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
        self.project_dets = tk.BooleanVar(value=True)
        # Perturbation state (only used when path_sens_adj is loaded)
        self.selected_medium = tk.IntVar(value=1)
        self.delta_mua = tk.DoubleVar(value=0.0)
        self.heatmap_mode = tk.StringVar(value="measurements")
        self._medium_cb: ttk.Combobox | None = None

        self.slice_photo: tk.PhotoImage | None = None
        self.heat_photo: tk.PhotoImage | None = None

        self.title(f"{output_dir.name} — adjoint sensor viewer")
        self.geometry("1300x780")
        self._build_ui()
        self.slice_canvas.bind("<Configure>", lambda _e: self._render_slice())
        self.heat_canvas.bind("<Configure>", lambda _e: self._render_heatmap())
        self.after(10, self._render_all)

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    def _det_idx(self) -> int:
        return self.det_row.get() * self.cols + self.det_col.get()

    def _product_volume(self) -> np.ndarray:
        idx = self._det_idx()
        vol = np.zeros(self.shape, dtype=np.float64)
        vol[self.marked_mask] = self.masked_forward * self.masked_adjoint[idx]
        return vol

    def _slice_array(self, vol: np.ndarray) -> tuple[np.ndarray, tuple[str, str]]:
        i = self.index.get()
        if self.axis.get() == "x":
            return vol[i, :, :].T, ("Y", "Z")
        if self.axis.get() == "y":
            return vol[:, i, :].T, ("X", "Z")
        return vol[:, :, i].T, ("X", "Y")

    def _near_slice(self, vox_pos: list[float], tolerance: float = 1.5) -> bool:
        axis_idx = "xyz".index(self.axis.get())
        return abs(float(vox_pos[axis_idx]) - self.index.get()) <= tolerance

    def _voxel_to_canvas(
        self, vox_pos: list[float], left: int, top: int, scale: int
    ) -> tuple[float, float]:
        """Map voxel (x,y,z) to canvas pixel for the current slice.

        Slice axes after transposition:
          axis=x: col=y, row=z
          axis=y: col=x, row=z
          axis=z: col=x, row=y
        """
        vx, vy, vz = float(vox_pos[0]), float(vox_pos[1]), float(vox_pos[2])
        axis = self.axis.get()
        if axis == "x":
            col, row = vy, vz
        elif axis == "y":
            col, row = vx, vz
        else:
            col, row = vx, vy
        return left + col * scale, top + row * scale

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
        ttk.Checkbutton(
            top, text="Project detectors", variable=self.project_dets, command=self._render_slice
        ).pack(side="left", padx=(6, 0))
        self.info_label = ttk.Label(top)
        self.info_label.pack(side="right")

        panes = ttk.PanedWindow(main, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(6, 0))

        self.slice_canvas = tk.Canvas(panes, background="#151515", highlightthickness=0)
        panes.add(self.slice_canvas, weight=3)
        self.slice_canvas.bind("<Button-1>", self._slice_click)

        right = ttk.Frame(panes, padding=(6, 0, 0, 0))
        panes.add(right, weight=1)

        heat_top = ttk.Frame(right)
        heat_top.pack(fill="x", pady=(0, 2))
        self.heat_title_label = ttk.Label(heat_top, text="Sensor measurements")
        self.heat_title_label.pack(side="left")
        if True:  # mode toggle always shown; sensitivity option requires path_sens_adj
            ttk.Radiobutton(
                heat_top, text="M", value="measurements",
                variable=self.heatmap_mode, command=self._heatmap_mode_changed,
            ).pack(side="right")
            ttk.Radiobutton(
                heat_top, text="S", value="sensitivity",
                variable=self.heatmap_mode, command=self._heatmap_mode_changed,
            ).pack(side="right", padx=(0, 4))
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

        # Legend
        legend = ttk.Label(right, text="● detectors  ✕ source  ■ selected", foreground="#888")
        legend.pack(anchor="w", pady=(2, 0))

        if self.path_sens_adj is not None and self.mua_orig is not None:
            self._build_perturbation_panel(right)

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

    def _slice_click(self, event: tk.Event) -> None:
        """Click on the slice view to select the tissue medium at that voxel."""
        if self.labels is None or self._medium_cb is None or self.mua_orig is None:
            return
        # Determine rendered image geometry (mirrors _render_slice)
        axis = self.axis.get()
        slices = {"x": self.shape[1], "y": self.shape[0], "z": self.shape[0]}
        W = slices.get(axis, 1)
        H = {"x": self.shape[2], "y": self.shape[2], "z": self.shape[1]}.get(axis, 1)
        if W == 0 or H == 0:
            return
        cw = max(1, self.slice_canvas.winfo_width())
        ch = max(1, self.slice_canvas.winfo_height())
        scale = max(1, min(cw // W, ch // H))
        left = (cw - W * scale) // 2
        top = (ch - H * scale) // 2
        col_i = (event.x - left) // scale
        row_i = (event.y - top) // scale
        if not (0 <= col_i < W and 0 <= row_i < H):
            return
        # Inverse of _slice_array (.T means col=fast axis, row=slow axis per slice)
        # axis=z: col=x, row=y;  axis=y: col=x, row=z;  axis=x: col=y, row=z
        idx = self.index.get()
        if axis == "z":
            vx, vy, vz = col_i, row_i, idx
        elif axis == "y":
            vx, vy, vz = col_i, idx, row_i
        else:
            vx, vy, vz = idx, col_i, row_i
        Nx, Ny, Nz = self.shape
        if not (0 <= vx < Nx and 0 <= vy < Ny and 0 <= vz < Nz):
            return
        label = int(self.labels[vx, vy, vz])
        if label <= 0 or label >= len(self.mua_orig):
            return
        cb_idx = label - 1  # combobox is 0-indexed over media 1..N
        if cb_idx < len(self._medium_cb["values"]):
            self._medium_cb.current(cb_idx)
            self._perturbation_changed()

    def _heatmap_mode_changed(self) -> None:
        mode = self.heatmap_mode.get()
        if mode == "sensitivity":
            if self.path_sens_adj is None:
                self.heatmap_mode.set("measurements")
                return
            self._update_sensitivity_title()
        else:
            self.heat_title_label.configure(text="Sensor measurements")
        self._render_heatmap()

    def _update_sensitivity_title(self) -> None:
        m = self.selected_medium.get()
        mua = float(self.mua_orig[m]) if self.mua_orig is not None else 0.0
        sens = self.path_sens_adj[:, m]
        nz = int(np.count_nonzero(sens))
        mn, mx = float(sens.min()), float(sens.max())
        self.heat_title_label.configure(
            text=f"Sensitivity  m{m} μa={mua:.4g}  [{mn:.3g}, {mx:.3g}]  nonzero={nz}/{len(sens)}"
        )

    def _heatmap_click(self, event: tk.Event) -> None:
        cw = max(1, self.heat_canvas.winfo_width())
        ch = max(1, self.heat_canvas.winfo_height())
        scale = min(max(1, cw // self.cols), max(1, ch // self.rows))
        img_w = self.cols * scale
        img_h = self.rows * scale
        left = (cw - img_w) // 2
        top = (ch - img_h) // 2
        col = (event.x - left) // scale
        row = (event.y - top) // scale
        self.det_col.set(max(0, min(self.cols - 1, int(col))))
        self.det_row.set(max(0, min(self.rows - 1, int(row))))
        self._render_all()

    # ------------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------------

    def _to_composite_rgb(
        self, labels_slice: np.ndarray | None, product_slice: np.ndarray
    ) -> bytes:
        """Labels as dim background, product as warm (orange) overlay at marked voxels."""
        H, W = product_slice.shape

        if labels_slice is not None and self.label_colors is not None:
            idx = np.clip(labels_slice.astype(np.int32), 0, len(self.label_colors) - 1)
            rgb = self.label_colors[idx].astype(np.float32)
        else:
            rgb = np.full((H, W, 3), 20.0, dtype=np.float32)

        vals = product_slice.astype(np.float64)
        positive = np.isfinite(vals) & (vals > 0)
        if np.any(positive):
            shown = vals.copy()
            if self.log_scale.get():
                floor = float(np.min(vals[positive]))
                shown[positive] = np.log10(vals[positive])
                shown[~positive] = np.log10(floor) - 1.0
                finite = np.isfinite(shown)
            else:
                finite = positive
            fvals = shown[finite]
            if self.autoscale.get():
                lo = float(np.percentile(fvals, 1.0))
                hi = float(np.percentile(fvals, 99.5))
            else:
                lo, hi = float(np.min(fvals)), float(np.max(fvals))
            if hi <= lo:
                hi = lo + 1.0
            t = np.zeros((H, W), dtype=np.float32)
            t[finite] = np.clip((shown[finite] - lo) / (hi - lo), 0.0, 1.0)
            warm = np.array([255.0, 200.0, 60.0], dtype=np.float32)
            alpha = (t * positive.astype(np.float32))[..., None]
            rgb = rgb * (1.0 - alpha) + warm * alpha

        return np.clip(rgb, 0, 255).astype(np.uint8).tobytes(order="C")

    def _to_heat_rgb(self, grid: np.ndarray) -> bytes:
        """Black → red → yellow thermal colormap for the measurements grid."""
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

    def _draw_cross(self, canvas: tk.Canvas, cx: float, cy: float, r: int, color: str) -> None:
        canvas.create_line(cx - r, cy, cx + r, cy, fill=color, width=2)
        canvas.create_line(cx, cy - r, cx, cy + r, fill=color, width=2)

    def _draw_circle(
        self, canvas: tk.Canvas, cx: float, cy: float, r: int, color: str, width: int = 2
    ) -> None:
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=width)

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def _render_slice(self) -> None:
        product_vol = self._product_volume()
        product_slice, ax_labels = self._slice_array(product_vol)

        labels_slice: np.ndarray | None = None
        if self.labels is not None:
            labels_slice, _ = self._slice_array(self.labels)

        H, W = product_slice.shape
        if W == 0 or H == 0:
            return

        pixels = self._to_composite_rgb(labels_slice, product_slice)
        photo = self._photo_from_pixels(W, H, pixels)

        cw = max(1, self.slice_canvas.winfo_width())
        ch = max(1, self.slice_canvas.winfo_height())
        scale = max(1, min(cw // W, ch // H))
        if scale > 1:
            photo = photo.zoom(scale, scale)
        self.slice_photo = photo

        img_w = W * scale
        img_h = H * scale
        left = (cw - img_w) // 2
        top = (ch - img_h) // 2

        self.slice_canvas.delete("all")
        self.slice_canvas.create_image(left, top, image=photo, anchor="nw")

        r = max(4, scale * 2)
        sel_row = self.det_row.get()
        sel_col = self.det_col.get()
        project = self.project_dets.get()

        # All non-selected detectors
        all_pos = self.det_positions.reshape(-1, 3)
        for det_idx, pos in enumerate(all_pos):
            pos_list = pos.tolist()
            row = det_idx // self.cols
            col = det_idx % self.cols
            if row == sel_row and col == sel_col:
                continue  # draw selected last (on top)
            if project or self._near_slice(pos_list):
                cx, cy = self._voxel_to_canvas(pos_list, left, top, scale)
                self._draw_circle(self.slice_canvas, cx, cy, r, "#888888")

        # Selected detector (on top, cyan, slightly larger)
        sel_pos = self.det_positions[sel_row, sel_col].tolist()
        if project or self._near_slice(sel_pos):
            cx, cy = self._voxel_to_canvas(sel_pos, left, top, scale)
            self._draw_circle(self.slice_canvas, cx, cy, r + 2, "#00c8ff", width=3)

        # Source cross (green, drawn last so it's always visible)
        if self.source_pos is not None and (project or self._near_slice(self.source_pos)):
            cx, cy = self._voxel_to_canvas(self.source_pos, left, top, scale)
            self._draw_cross(self.slice_canvas, cx, cy, r + 2, "#00ff44")

        positive = product_slice[np.isfinite(product_slice) & (product_slice > 0)]
        max_val = float(np.max(positive)) if positive.size else math.nan
        meas = float(self.measurements[self._det_idx()])
        axis = self.axis.get()
        ax_size = self.shape["xyz".index(axis)]
        self.info_label.configure(
            text=f"det ({sel_row},{sel_col})  "
                 f"{axis.upper()} {self.index.get() + 1}/{ax_size}  "
                 f"slice max {max_val:.4g}  M {meas:.4g}"
        )

    def _render_heatmap(self) -> None:
        if self.heatmap_mode.get() == "sensitivity" and self.path_sens_adj is not None:
            m = self.selected_medium.get()
            meas = self.path_sens_adj[:, m]
        else:
            meas = self._perturbed_measurements()
        pixels = self._to_heat_rgb(meas.reshape(self.rows, self.cols))
        photo = self._photo_from_pixels(self.cols, self.rows, pixels)

        cw = max(1, self.heat_canvas.winfo_width())
        ch = max(1, self.heat_canvas.winfo_height())
        scale = min(max(1, cw // self.cols), max(1, ch // self.rows))
        if scale > 1:
            photo = photo.zoom(scale, scale)
        self.heat_photo = photo

        img_w = self.cols * scale
        img_h = self.rows * scale
        left = (cw - img_w) // 2
        top = (ch - img_h) // 2

        self.heat_canvas.delete("all")
        self.heat_canvas.create_image(left, top, image=photo, anchor="nw")

        sel_row = self.det_row.get()
        sel_col = self.det_col.get()
        x0 = left + sel_col * scale
        y0 = top + sel_row * scale
        self.heat_canvas.create_rectangle(
            x0, y0, x0 + scale, y0 + scale, outline="#00c8ff", width=2
        )

        pos = self.det_positions[sel_row, sel_col]
        det_meas = float(meas[self._det_idx()])
        self.det_meas_label.configure(
            text=f"pos [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]  M = {det_meas:.4g}"
        )

    # ------------------------------------------------------------------
    # perturbation panel
    # ------------------------------------------------------------------

    def _build_perturbation_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="μa perturbation (adjoint side)", padding=4)
        frame.pack(fill="x", pady=(8, 0))

        ttk.Label(frame, text="Medium:").grid(row=0, column=0, sticky="w")
        media_values = [
            f"{m}: μa={self.mua_orig[m]:.4g} mm⁻¹"
            for m in range(1, len(self.mua_orig))
        ]
        self._medium_cb = ttk.Combobox(frame, values=media_values, state="readonly", width=22)
        self._medium_cb.current(0)
        self._medium_cb.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._medium_cb.bind("<<ComboboxSelected>>", lambda _e: self._perturbation_changed())

        ttk.Label(frame, text="Δμa:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        mua1 = float(self.mua_orig[1]) if len(self.mua_orig) > 1 else 0.1
        self._delta_slider = ttk.Scale(
            frame,
            orient="horizontal",
            variable=self.delta_mua,
            from_=-mua1,
            to=2.0 * mua1,
            command=lambda _: self._perturbation_changed(),
        )
        self._delta_slider.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))

        self._perturb_label = ttk.Label(frame, text="Δμa = 0.0000 mm⁻¹")
        self._perturb_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._perturb_effect_label = ttk.Label(frame, text="")
        self._perturb_effect_label.grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Button(frame, text="Reset Δμa", command=self._reset_perturbation).grid(
            row=4, column=0, columnspan=2, pady=(4, 0)
        )
        frame.columnconfigure(1, weight=1)

    def _perturbation_changed(self) -> None:
        m = self._medium_cb.current() + 1  # 1-based medium index
        self.selected_medium.set(m)
        delta = self.delta_mua.get()
        mua_new = float(self.mua_orig[m]) + delta
        self._perturb_label.configure(
            text=f"Δμa = {delta:+.4f} mm⁻¹   μa_new = {mua_new:.4g} mm⁻¹"
        )
        perturbed = self._perturbed_measurements()
        rel = (perturbed - self.measurements) / (np.abs(self.measurements) + 1e-30)
        max_pct = float(np.max(np.abs(rel))) * 100.0
        self._perturb_effect_label.configure(text=f"max |ΔM/M| = {max_pct:.1f}%")
        if self.heatmap_mode.get() == "sensitivity":
            self._update_sensitivity_title()
        self._render_heatmap()

    def _reset_perturbation(self) -> None:
        self.delta_mua.set(0.0)
        self._perturbation_changed()

    def _perturbed_measurements(self) -> np.ndarray:
        if self.path_sens_adj is None:
            return self.measurements
        m = self.selected_medium.get()
        delta = self.delta_mua.get()
        if delta == 0.0:
            return self.measurements
        return self.measurements - delta * self.path_sens_adj[:, m]

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

    # Load tissue labels and source position from base config
    labels: np.ndarray | None = None
    source_pos: list[float] | None = None
    base_config_path = Path(meta.get("base_config", ""))
    if base_config_path.exists():
        try:
            base_config = load_json_relaxed(base_config_path)
        except Exception as exc:
            print(f"warning: could not load base config: {exc}")
            base_config = {}
        try:
            labels = decode_shapes(base_config)
        except Exception as exc:
            print(f"warning: could not decode tissue labels: {exc}")
        try:
            pos = base_config["Optode"]["Source"]["Pos"]
            source_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        except Exception:
            pass

    # Load per-medium path sensitivity if available (requires DoPartialPath run)
    path_sens_adj: np.ndarray | None = None
    mua_orig: np.ndarray | None = None
    path_sens_path = output_dir / "path_sens_adj.npy"
    if path_sens_path.exists():
        path_sens_adj = np.load(path_sens_path)
        mua_list = meta.get("mua_orig")
        if mua_list:
            mua_orig = np.array(mua_list, dtype=np.float64)
    else:
        print("note: path_sens_adj.npy not found — perturbation panel disabled")

    viewer = SensorViewer(
        output_dir,
        masked_forward,
        masked_adjoint,
        marked_mask,
        det_positions,
        rows,
        cols,
        labels=labels,
        source_pos=source_pos,
        path_sens_adj=path_sens_adj,
        mua_orig=mua_orig,
    )
    viewer.mainloop()


if __name__ == "__main__":
    main()
