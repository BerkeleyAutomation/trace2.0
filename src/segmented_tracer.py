"""
segmented_tracer.py  –  Segmented cable tracing pipeline.

Divides the cable mask image into overlapping spatial tiles, runs the learned
tracer independently on each tile, stitches the per-tile traces into one
continuous full-image trace per cable, then runs divergence analysis.

Key idea
--------
The existing tracer already uses a local 256×256 crop (autoregressive),
but it reasons about a single *continuous* path from start to finish.
This module instead *explicitly partitions* the image into N×M tiles and
processes each tile as an independent tracing problem:

    Full image
        │
        ├─ Tile (0,0) ──> trace segment 0
        ├─ Tile (0,1) ──> trace segment 1   (started from exit of seg 0)
        ├─ Tile (1,0) ──> trace segment 2   (started from exit of seg 1)
        └─ …
                └─ stitch ──> continuous cable trace

Coordinate conventions
----------------------
  • "half-res"  – the resolution the tracer and endpoints work in
                  (endpoints detected on img_down = full/2; get_mask returns
                  half-res)
  • "full-res"  – the raw camera image; the Detic bool mask lives here,
                  accessed by the tracer as  bool_detic_mask[row*2, col*2]
  • Tile bounds are stored in half-res coordinates.
  • The full-res detic mask crop uses  mask[r0*2:r1*2, c0*2:c1*2].

Thread safety
-------------
Multiple endpoint traces run in parallel (one thread per endpoint).
Inside each endpoint thread the tile loop is sequential.  A threading.Lock
protects the brief window where the shared Tracer's bool_detic_mask /
detic_mask attributes are swapped for tile-local crops.

Live visualisation
------------------
A ProgressVisualizer is created before tracing starts.  Each thread calls
it when a tile trace begins (highlights that tile) and when it finishes
(draws the segment).  The canvas is written to  live_progress.png  after
every update so you can monitor it even without a display.  Pass --viz to
also open an interactive cv2 window.

Usage
-----
    python segmented_tracer.py --image input.png --tier 2
    python segmented_tracer.py --image input.png --tier 4 \\
        --grid_rows 3 --grid_cols 3 --overlap 0.3 --path_len 80 --viz
"""

TIER_TO_ENDPOINTS = {1: 4, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16}

import argparse
import copy
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.detic_dataloader import DeticDataloader
from utils.tracer_dataloader import TracerDataloader
from utils.endpoint_detector_dataloader import EndpointDataloader
from utils.tracer.tusk_pipeline.tracer import TraceEnd
from divergencecopy import get_all_div_points, visualize_multiple_paths
from masker import get_mask
from run_constants import COLORS, SAVE_DIR

# Lock protecting temporary mutation of the shared Tracer's detic-mask attrs.
_TRACER_MASK_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers duplicated from main_vision.py to keep this file self-contained
# ─────────────────────────────────────────────────────────────────────────────

def _dilate_masks(mask_tensor, kernel_size=3):
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.float()
    dilated = F.max_pool2d(mask_tensor, kernel_size, stride=1,
                           padding=kernel_size // 2)
    return dilated.squeeze()


def _get_detic_masks(img_rgb, detic):
    out = detic.predict(img_rgb)
    keep = [out['boxes'][i][0] > 100 and out['boxes'][i][2] < 3500
            for i in range(out['boxes'].shape[0])]
    out['boxes'] = out['boxes'][keep]
    out['masks'] = out['masks'][keep]
    out['masks'] = detic.filter_mask_by_area(out['masks'], max_area=250_000)

    h, w = out['masks'].shape[2], out['masks'].shape[3]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    detic_arr = torch.zeros((h, w), dtype=torch.uint8).to(dev)
    bool_arr  = torch.zeros((h, w), dtype=torch.bool).to(dev)
    for i in range(out['masks'].shape[0]):
        for j in range(out['masks'].shape[1]):
            detic_arr = torch.maximum(detic_arr, out['masks'][i][j] * (i + 1))
            bool_arr  = torch.logical_or(bool_arr, out['masks'][i][j])
    return bool_arr, detic_arr


def _predict_endpoints(img_rgb, endpt_model, bool_detic_mask=None):
    img_down = cv2.resize(img_rgb, (img_rgb.shape[1] // 2, img_rgb.shape[0] // 2))
    endpoints = endpt_model.predict(img_down)['endpoints']
    if bool_detic_mask is not None:
        endpoints = [ep for ep in endpoints
                     if bool_detic_mask[ep[0] * 2][ep[1] * 2] == 0]
    return endpoints


# ─────────────────────────────────────────────────────────────────────────────
# Live progress visualizer
# ─────────────────────────────────────────────────────────────────────────────

def _density_color(density):
    """Map density 0-1 → RGB: green → yellow → red."""
    r = min(255, int(density * 2 * 255))
    g = min(255, int((1.0 - density) * 2 * 255))
    return (r, g, 0)


class ProgressVisualizer:
    """
    Thread-safe live visualizer.  Redraws the canvas whenever a tile starts,
    a segment finishes, or density data is updated.  Saves live_progress.png
    after every change.  Pass show_window=True to also open a cv2 window.

    Colour conventions
    ------------------
    • Light-grey thin rectangle  – tile boundary (always shown)
    • Coloured tile fill         – cable density heatmap (green=low, red=high)
    • Thick coloured rectangle   – tile currently being traced by that endpoint
    • Coloured line              – completed trace segment
    • Green circle               – segment start point
    • Blue circle                – EDGE end (trace continues in next tile)
    • Red circle                 – terminal end (ENDPOINT / OBJECT / RETRACE)
    • Yellow dot                 – detected endpoint
    • Orange double-border + brackets + "FOCUS" label – highest-density tile
      waiting to be resolved (the attention / priority frame)
    """

    def __init__(self, img_halfres, tiles, endpoints, output_dir,
                 show_window=False):
        self._base           = img_halfres.copy()
        self._tiles          = tiles
        self._tiles_by_key   = {(t['row'], t['col']): t for t in tiles}
        self._out            = output_dir
        self._lock           = threading.Lock()
        self._show           = show_window
        self._segments       = {}   # ep_idx -> list of (seg_xy, trace_end)
        self._active         = {}   # ep_idx -> tile dict
        self._tile_densities = {}   # (row,col) -> float 0-1
        self._attention_key  = None # (row,col) of the focus tile, or None
        self._endpoints      = endpoints

        self._base_with_grid = self._build_base(img_halfres, tiles, endpoints)
        self._save_and_show(self._base_with_grid)

        if show_window:
            cv2.namedWindow('Segmented Tracer – live', cv2.WINDOW_NORMAL)
            cv2.imshow('Segmented Tracer – live',
                       cv2.cvtColor(self._base_with_grid, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    # ── public thread-safe API ─────────────────────────────────────────────

    def on_tile_start(self, ep_idx, tile):
        """Call when a thread begins tracing a tile."""
        with self._lock:
            self._active[ep_idx] = tile
            self._redraw()

    def on_segment_done(self, ep_idx, seg_xy, trace_end):
        """Call when a tile-trace finishes and the segment is ready."""
        with self._lock:
            self._active.pop(ep_idx, None)
            self._segments.setdefault(ep_idx, []).append((seg_xy, trace_end))
            self._redraw()

    def update_densities(self, densities):
        """
        Update per-tile density values and redraw.

        Parameters
        ----------
        densities : dict  (row, col) -> float 0-1
        """
        with self._lock:
            self._tile_densities = dict(densities)
            self._redraw()

    def set_attention_tile(self, tile_key):
        """
        Highlight tile_key=(row,col) with the orange attention/focus frame.
        Pass None to clear the attention frame.
        """
        with self._lock:
            self._attention_key = tile_key
            self._redraw()

    def update_base_image(self, img_halfres):
        """Swap in a new background image (e.g. after robot moved)."""
        with self._lock:
            self._base_with_grid = self._build_base(
                img_halfres, self._tiles, self._endpoints)
            self._redraw()

    def close(self):
        """Wait for a keypress then destroy window."""
        if self._show:
            print("Press any key in the visualiser window to close…")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_base(img_halfres, tiles, endpoints):
        base = img_halfres.copy()
        for t in tiles:
            cv2.rectangle(base,
                          (t['c0'], t['r0']), (t['c1'], t['r1']),
                          (160, 160, 160), 1)
            cv2.putText(base, f"({t['row']},{t['col']})",
                        (t['c0'] + 4, t['r0'] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        for ep in endpoints:
            cv2.circle(base, (int(ep[1]), int(ep[0])), 6, (255, 220, 0), -1)
        return base

    def _redraw(self):
        """Rebuild canvas from scratch.  Must be called with self._lock held."""
        canvas = self._base_with_grid.copy()

        # ── 1. Density heatmap overlay (semi-transparent fills) ───────────
        if self._tile_densities:
            overlay = canvas.copy()
            for key, density in self._tile_densities.items():
                t = self._tiles_by_key.get(key)
                if t is None:
                    continue
                color = _density_color(density)
                cv2.rectangle(overlay,
                              (t['c0'], t['r0']), (t['c1'], t['r1']),
                              color, -1)
                # Density % label at tile centre
                cx = (t['c0'] + t['c1']) // 2 - 18
                cy = (t['r0'] + t['r1']) // 2
                cv2.putText(overlay, f"{density:.0%}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(overlay, f"{density:.0%}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            canvas = cv2.addWeighted(canvas, 0.65, overlay, 0.35, 0)

        # ── 2. Completed trace segments ───────────────────────────────────
        for ep_idx, segs in self._segments.items():
            color = COLORS[ep_idx % len(COLORS)]
            for seg_xy, trace_end in segs:
                if len(seg_xy) < 2:
                    continue
                for i in range(len(seg_xy) - 1):
                    cv2.line(canvas,
                             tuple(seg_xy[i].astype(int)),
                             tuple(seg_xy[i + 1].astype(int)),
                             color, 2)
                cv2.circle(canvas, tuple(seg_xy[0].astype(int)),
                           4, (0, 220, 0), -1)
                end_color = (50, 50, 255) if trace_end == TraceEnd.EDGE \
                            else (220, 0, 0)
                cv2.circle(canvas, tuple(seg_xy[-1].astype(int)),
                           4, end_color, -1)

        # ── 3. Active-tracing tile borders ────────────────────────────────
        for ep_idx, tile in self._active.items():
            color = COLORS[ep_idx % len(COLORS)]
            cv2.rectangle(canvas,
                          (tile['c0'], tile['r0']),
                          (tile['c1'], tile['r1']),
                          color, 3)
            cv2.putText(canvas, f"ep{ep_idx}…",
                        (tile['c0'] + 4, tile['r0'] + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── 4. Attention / focus frame on highest-density tile ────────────
        if self._attention_key is not None:
            t = self._tiles_by_key.get(self._attention_key)
            if t is not None:
                r0, r1, c0, c1 = t['r0'], t['r1'], t['c0'], t['c1']
                ORANGE = (255, 140,  0)
                YELLOW = (255, 255,  0)
                # Outer thick orange border
                cv2.rectangle(canvas, (c0, r0), (c1, r1), ORANGE, 4)
                # Inner yellow border
                cv2.rectangle(canvas, (c0+5, r0+5), (c1-5, r1-5), YELLOW, 2)
                # Corner brackets (L-shapes at each corner)
                cs = 18   # corner segment length
                for (cy, cx), (dy, dx) in [
                    ((r0, c0), ( 1,  1)), ((r0, c1), ( 1, -1)),
                    ((r1, c0), (-1,  1)), ((r1, c1), (-1, -1)),
                ]:
                    cv2.line(canvas, (cx, cy), (cx + dx*cs, cy), YELLOW, 3)
                    cv2.line(canvas, (cx, cy), (cx, cy + dy*cs), YELLOW, 3)
                # "FOCUS" label with density
                density = self._tile_densities.get(self._attention_key, 0.0)
                label = f"FOCUS  {density:.0%}"
                lx, ly = c0 + 6, r0 + 52
                cv2.putText(canvas, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(canvas, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

        # ── 5. Status bar ─────────────────────────────────────────────────
        n_done   = sum(len(v) for v in self._segments.values())
        n_active = len(self._active)
        attn_str = (f"  |  attention: {self._attention_key}"
                    if self._attention_key else "")
        status = (f"tiles done: {n_done}  |  active: {n_active}"
                  f"  |  eps: {len(self._segments)}{attn_str}")
        cv2.putText(canvas, status, (8, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)

        self._save_and_show(canvas)

    def _save_and_show(self, canvas):
        cv2.imwrite(os.path.join(self._out, 'live_progress.png'),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if self._show:
            cv2.imshow('Segmented Tracer – live',
                       cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


# ─────────────────────────────────────────────────────────────────────────────
# Tile management
# ─────────────────────────────────────────────────────────────────────────────

def make_tiles(img_h, img_w, grid_rows, grid_cols, overlap_frac=0.25):
    """
    Partition (img_h × img_w) into a grid_rows × grid_cols grid with
    fractional overlap.  Coordinates are in *half-res* (tracer) space.

    Returns a list of dicts: {row, col, r0, r1, c0, c1}.
    """
    base_h = img_h / grid_rows
    base_w = img_w / grid_cols
    pad_h  = int(base_h * overlap_frac)
    pad_w  = int(base_w * overlap_frac)

    tiles = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            tiles.append(dict(
                row=r, col=c,
                r0=max(0,     int(r       * base_h) - pad_h),
                r1=min(img_h, int((r + 1) * base_h) + pad_h),
                c0=max(0,     int(c       * base_w) - pad_w),
                c1=min(img_w, int((c + 1) * base_w) + pad_w),
            ))
    return tiles


def _tile_for_point(pt_rc, tiles):
    """Return the first tile that contains pt_rc=(row, col)."""
    for t in tiles:
        if t['r0'] <= pt_rc[0] < t['r1'] and t['c0'] <= pt_rc[1] < t['c1']:
            return t
    # Fall back to tile whose centre is closest (handles exact-edge cases).
    return min(tiles, key=lambda t: np.hypot(
        pt_rc[0] - (t['r0'] + t['r1']) / 2,
        pt_rc[1] - (t['c0'] + t['c1']) / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Single-tile tracing
# ─────────────────────────────────────────────────────────────────────────────

def _trace_in_tile(tile,
                   img_mask_halfres_rgb,
                   bool_detic_full, detic_full,
                   start_pts_global, endpoints_global,
                   learned_tracer, path_len):
    """
    Crop mask + detic masks to *tile*, convert coordinates to tile-local space,
    run the learned tracer, then convert results back to global (half-res).

    Because the tile image is passed as the tracer input, TraceEnd.EDGE fires
    naturally when the trace reaches the tile boundary (the tracer's img.shape
    boundary check fires against the tile dimensions).

    Returns
    -------
    seg_global  : np.ndarray (N, 2) in global half-res (row, col)
    trace_end   : TraceEnd
    densities   : list[float]
    mask_num    : int  (hit object index, or -1)
    """
    r0, r1, c0, c1 = tile['r0'], tile['r1'], tile['c0'], tile['c1']
    offset = np.array([r0, c0])

    # Crop cable mask to this tile (half-res bounds).
    tile_mask = img_mask_halfres_rgb[r0:r1, c0:c1]

    # Crop Detic masks: full-res bounds = 2 × half-res tile bounds.
    bdem_tile = bool_detic_full[r0 * 2:r1 * 2, c0 * 2:c1 * 2]
    dem_tile  = detic_full[r0 * 2:r1 * 2, c0 * 2:c1 * 2].clone()

    # Convert start_pts from global → tile-local.
    start_pts_local = np.array(start_pts_global) - offset

    # Convert endpoints from global → tile-local; keep only those inside tile.
    endpoints_local = []
    for ep in endpoints_global:
        ep_l = np.array(ep) - offset
        if 0 <= ep_l[0] < (r1 - r0) and 0 <= ep_l[1] < (c1 - c0):
            endpoints_local.append(ep_l)
    endpoints_local = endpoints_local if endpoints_local else None

    # Swap Tracer's detic masks for tile-local versions, trace, then restore.
    # The lock serialises this across endpoint threads.
    with _TRACER_MASK_LOCK:
        orig_bool = learned_tracer.bool_detic_mask
        orig_det  = learned_tracer.detic_mask
        learned_tracer.bool_detic_mask = bdem_tile
        learned_tracer.detic_mask      = dem_tile
        try:
            output = learned_tracer.trace(
                tile_mask, start_pts_local, endpoints_local,
                path_len=path_len, use_vit=False)
        finally:
            learned_tracer.bool_detic_mask = orig_bool
            learned_tracer.detic_mask      = orig_det

    # Convert tile-local (row, col) → global (row, col).
    seg_global = np.array(output['trace']) + offset
    return seg_global, output['trace_end'], output['densities'], output['mask_num']


# ─────────────────────────────────────────────────────────────────────────────
# Per-endpoint segmented trace
# ─────────────────────────────────────────────────────────────────────────────

def _trace_endpoint_segmented(ep_idx, img_full, endpts,
                               learned_tracer, analytic_tracer,
                               bool_detic_full, detic_full,
                               grid_rows, grid_cols,
                               overlap_frac, path_len_per_tile,
                               visualizer=None):
    """
    Trace a single cable from endpoint *ep_idx* using tile-by-tile tracing.

    Returns
    -------
    dict with keys:
        ep_idx, trace_path (x,y for visualisation), endpt_entry,
        density_norm, mask_num, segments (list of per-tile x,y arrays)
    """
    condition_len = learned_tracer.trace_config.condition_len

    # Build full-image cable mask (half-res output).
    img_mask = get_mask(img_full, endpts[ep_idx])
    img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

    H2, W2 = img_mask_rgb.shape[:2]
    tiles = make_tiles(H2, W2, grid_rows, grid_cols, overlap_frac)

    # Seed the first tile with the analytic tracer (global half-res coords).
    filtered_endpts = copy.deepcopy(endpts)
    filtered_endpts[ep_idx] = np.array([-100, -100])

    start_pts, _ = analytic_tracer.trace(
        img_mask_rgb, np.array(endpts[ep_idx]), path_len=3)
    start_pts = (np.array(start_pts)
                 if start_pts is not None
                 else np.empty((0, 2)))

    if len(start_pts) < condition_len:
        print(f"[seg] ep {ep_idx}: analytic gave {len(start_pts)} pts "
              f"(need {condition_len}), returning short trace")
        trace_path = (np.flip(start_pts, axis=1)
                      if start_pts.size
                      else np.array([endpts[ep_idx][::-1]]))
        return dict(ep_idx=ep_idx, trace_path=trace_path,
                    endpt_entry=[ep_idx, -1], density_norm=np.array([]),
                    mask_num=-1, segments=[])

    # ── tile-by-tile loop ─────────────────────────────────────────────────
    segments_global  = []   # list of np.ndarray (N,2), global half-res (row,col)
    all_densities    = []
    final_trace_end  = TraceEnd.FINISHED
    final_mask_num   = -1
    visited          = set()
    current_seed     = start_pts   # global half-res (row, col)

    for _ in range(len(tiles) + 1):   # +1 as safety cap
        head = current_seed[-1]
        tile = _tile_for_point(head, tiles)
        key  = (tile['row'], tile['col'])

        if key in visited:
            print(f"[seg] ep {ep_idx}: revisiting tile {key} – stopping")
            break
        visited.add(key)

        print(f"[seg] ep {ep_idx}: tile {key} "
              f"rows[{tile['r0']}:{tile['r1']}] cols[{tile['c0']}:{tile['c1']}]")

        if visualizer is not None:
            visualizer.on_tile_start(ep_idx, tile)

        seg, trace_end, densities, mask_num = _trace_in_tile(
            tile, img_mask_rgb,
            bool_detic_full, detic_full,
            current_seed, filtered_endpts,
            learned_tracer, path_len_per_tile)

        # Flip (row,col)→(x,y) for the visualizer before reporting.
        if visualizer is not None and len(seg) > 0:
            visualizer.on_segment_done(ep_idx, np.flip(seg, axis=1), trace_end)

        if len(seg) > 0:
            segments_global.append(seg)
        all_densities.extend(densities)
        final_trace_end = trace_end
        final_mask_num  = mask_num

        # Non-EDGE endings are terminal (ENDPOINT / OBJECT / RETRACE / FINISHED).
        if trace_end != TraceEnd.EDGE:
            break

        # Seed the next tile from the tail of this segment.
        if len(seg) >= condition_len:
            current_seed = seg[-condition_len:]
        else:
            print(f"[seg] ep {ep_idx}: segment too short to continue")
            break

    # ── stitch ───────────────────────────────────────────────────────────
    full_trace = (np.vstack(segments_global)
                  if segments_global
                  else np.array([endpts[ep_idx]]))

    # ── endpoint table entry ──────────────────────────────────────────────
    endpt_entry = [ep_idx, -1]
    if final_trace_end == TraceEnd.ENDPOINT and len(full_trace) > 0:
        last = full_trace[-1]
        for j, ep in enumerate(filtered_endpts):
            if np.linalg.norm(last - ep) < learned_tracer.ep_buffer:
                endpt_entry = [ep_idx, j]
                break

    # ── normalise densities ───────────────────────────────────────────────
    dens = np.array(all_densities)
    if dens.size > 0:
        rng = dens.max() - dens.min()
        density_norm = (dens - dens.min()) / rng if rng > 0 else np.zeros_like(dens)
    else:
        density_norm = np.array([])

    # Flip (row, col) → (col, row) = (x, y) for visualisation, matching the
    # convention in main_vision.py / divergencecopy.py.
    trace_path_xy = np.flip(full_trace, axis=1)
    segments_xy   = [np.flip(s, axis=1) for s in segments_global]

    total_pts = sum(len(s) for s in segments_global)
    print(f"[seg] ep {ep_idx}: {len(segments_global)} tile(s), "
          f"{total_pts} pts total, end={final_trace_end.name}")

    return dict(ep_idx=ep_idx, trace_path=trace_path_xy,
                endpt_entry=endpt_entry, density_norm=density_norm,
                mask_num=final_mask_num, segments=segments_xy)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel wrapper  (one thread per cable endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def get_segmented_trace_list(img_full, endpts, tracer_wrapper,
                              bool_detic_full, detic_full,
                              grid_rows=2, grid_cols=2,
                              overlap_frac=0.25, path_len_per_tile=120,
                              visualizer=None):
    """
    Trace all endpoints in parallel (one thread per endpoint) using the
    segmented approach.

    Parameters
    ----------
    img_full         : np.ndarray (H, W, 3)  full-resolution RGB image
    endpts           : list of [row, col] in half-res
    tracer_wrapper   : TracerDataloader instance
    bool_detic_full  : dilated boolean Detic mask (full-res, on CUDA)
    detic_full       : dilated indexed Detic mask  (full-res, on CUDA)
    grid_rows/cols   : tile grid dimensions
    overlap_frac     : fractional overlap between tiles (0–1)
    path_len_per_tile: max tracer steps per tile
    visualizer       : ProgressVisualizer or None

    Returns
    -------
    trace_list      : list of (N,2) arrays in (x,y) for visualisation
    endpt_table     : list of [ep_idx, matched_ep_idx]
    densities_nrm   : list of 1-D density arrays (one per endpoint)
    mask_hit        : first hit object index, or -1
    segments_by_ep  : list of lists – per-ep list of per-tile (x,y) traces
    """
    learned  = tracer_wrapper.tracer
    analytic = tracer_wrapper.analytic_tracer
    L        = len(endpts)

    results = {}
    with ThreadPoolExecutor(max_workers=L) as ex:
        futs = {
            ex.submit(
                _trace_endpoint_segmented,
                ep_idx, img_full, endpts,
                learned, analytic,
                bool_detic_full, detic_full,
                grid_rows, grid_cols, overlap_frac, path_len_per_tile,
                visualizer
            ): ep_idx
            for ep_idx in range(L)
        }
        for fut in as_completed(futs):
            r = fut.result()
            results[r['ep_idx']] = r

    trace_list, endpt_table, densities_nrm, segments_by_ep = [], [], [], []
    mask_hits = []
    for ep_idx in range(L):
        r = results[ep_idx]
        trace_list.append(r['trace_path'])
        endpt_table.append(r['endpt_entry'])
        densities_nrm.append(r['density_norm'])
        segments_by_ep.append(r['segments'])
        mask_hits.append(r['mask_num'])

    hit_arr = np.array([int(m) - 1 for m in mask_hits if m != -1])
    valid    = np.argwhere(hit_arr > -1)
    mask_hit = int(hit_arr[valid[0]]) if len(valid) > 0 else -1

    return trace_list, endpt_table, densities_nrm, mask_hit, segments_by_ep


# ─────────────────────────────────────────────────────────────────────────────
# Final visualisation saves
# ─────────────────────────────────────────────────────────────────────────────

def save_tile_grid_image(img_halfres, tiles, output_dir):
    """Annotate image with tile bounding boxes and save."""
    vis = img_halfres.copy()
    for t in tiles:
        cv2.rectangle(vis, (t['c0'], t['r0']), (t['c1'], t['r1']),
                      color=(255, 0, 0), thickness=2)
        cv2.putText(vis, f"({t['row']},{t['col']})",
                    (t['c0'] + 5, t['r0'] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    path = os.path.join(output_dir, 'tile_grid.png')
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved: {path}")


def save_per_segment_images(img_halfres, segments_by_ep, output_dir):
    """
    One image per endpoint showing each tile-segment in a different colour.
    Green = segment start, red = terminal end, blue = EDGE (continued to next tile).
    """
    for ep_idx, segments in enumerate(segments_by_ep):
        if not segments:
            continue
        vis = img_halfres.copy()
        for seg_i, seg in enumerate(segments):
            color = COLORS[seg_i % len(COLORS)]
            for i in range(len(seg) - 1):
                cv2.line(vis, tuple(seg[i].astype(int)),
                         tuple(seg[i + 1].astype(int)), color, 2)
            if len(seg) > 0:
                cv2.circle(vis, tuple(seg[0].astype(int)),  5, (0, 220, 0), -1)
                cv2.circle(vis, tuple(seg[-1].astype(int)), 5, (220, 0, 0), -1)
        path = os.path.join(output_dir, f'segments_ep{ep_idx}.png')
        cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved: {path}")


def save_combined_trace_image(img_halfres, trace_list, output_dir, viz=False):
    """All stitched traces on one image."""
    combined = visualize_multiple_paths(
        img_halfres.copy(), trace_list,
        [COLORS[i % len(COLORS)] for i in range(len(trace_list))])
    path = os.path.join(output_dir, 'segmented_traces_combined.png')
    cv2.imwrite(path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Saved: {path}")
    if viz:
        plt.figure(figsize=(16, 12))
        plt.imshow(combined)
        plt.title(f'Segmented Traces – {len(trace_list)} cables')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_segmented_pipeline(image_path, output_dir, num_endpoints,
                            grid_rows=2, grid_cols=2,
                            overlap_frac=0.25, path_len_per_tile=120,
                            viz=False):
    """
    End-to-end segmented vision pipeline on a static image.

    Stages
    ------
    1. Detic object detection  (full-res, sequential)
    2. Endpoint detection      (full-res, sequential)
    3. Segmented parallel tracing  (half-res tiles, parallel per endpoint)
    4. Divergence analysis
    5. Save visualisations

    Returns
    -------
    dict: trace_list, endpoints, div_output, segments_by_ep
    """
    print(f"Loading: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load: {image_path}")
    img_rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_down = cv2.resize(img_rgb, (img_rgb.shape[1] // 2, img_rgb.shape[0] // 2))

    print("Initialising models…")
    endpt_model  = EndpointDataloader(use_hub_detect=True)
    detic_loader = DeticDataloader()
    detic_loader.create()
    detic_loader.default_vocab()
    tracer = TracerDataloader()
    print("Models ready.")

    t0 = time.time()

    # ── 1 + 2: Detic and endpoint detection ────────────────────────────────
    bool_detic, detic_mask = _get_detic_masks(img_rgb, detic_loader)
    img_gray = cv2.convertScaleAbs(
        cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
                     cv2.COLOR_GRAY2RGB),
        alpha=1.5, beta=-20)
    endpoints = _predict_endpoints(img_gray, endpt_model, bool_detic)
    print(f"  {len(endpoints)} endpoints detected (expected {num_endpoints})")
    if len(endpoints) != num_endpoints:
        print("  WARNING: endpoint count mismatch")

    bool_detic_dil = _dilate_masks(bool_detic,    15)
    detic_mask_dil = _dilate_masks(detic_mask,    15).clone()

    t1 = time.time()
    print(f"Detection: {t1 - t0:.1f}s")

    # ── 3: Build tile layout and create live visualizer ────────────────────
    H2, W2 = img_down.shape[:2]
    tiles = make_tiles(H2, W2, grid_rows, grid_cols, overlap_frac)

    visualizer = ProgressVisualizer(
        img_down, tiles, endpoints, output_dir,
        show_window=viz)

    print(f"Segmented tracing  grid={grid_rows}×{grid_cols}  "
          f"overlap={overlap_frac:.0%}  path_len/tile={path_len_per_tile}")
    print(f"Live progress image: {os.path.join(output_dir, 'live_progress.png')}")

    trace_list, endpt_table, densities, _, segments_by_ep = \
        get_segmented_trace_list(
            img_rgb, endpoints, tracer,
            bool_detic_dil, detic_mask_dil,
            grid_rows=grid_rows, grid_cols=grid_cols,
            overlap_frac=overlap_frac, path_len_per_tile=path_len_per_tile,
            visualizer=visualizer)

    t2 = time.time()
    print(f"Tracing: {t2 - t1:.1f}s   total: {t2 - t0:.1f}s")

    # ── 4: Divergence analysis ──────────────────────────────────────────────
    print("Finding divergence points…")
    div_output = get_all_div_points(
        img_down, endpoints, trace_list, endpt_table, densities)
    n_div = len(div_output.get('div_points', []))
    print(f"  {n_div} divergence point(s) found")

    # ── 5: Save final visualisations ───────────────────────────────────────
    save_tile_grid_image(img_down, tiles, output_dir)
    save_per_segment_images(img_down, segments_by_ep, output_dir)
    save_combined_trace_image(img_down, trace_list, output_dir, viz)

    visualizer.close()

    print(f"\nOutput saved to: {output_dir}")
    return dict(trace_list=trace_list, endpoints=endpoints,
                div_output=div_output, segments_by_ep=segments_by_ep)


# ─────────────────────────────────────────────────────────────────────────────
# Tile density computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_tile_cable_density(img_halfres, tiles, cable_thresh=80):
    """
    Compute the fraction of cable-like pixels in each tile.

    Uses luminance: cable pixels are assumed to be bright (white/coloured)
    against a dark background.  Operates directly on the half-res image so
    no extra resizing is needed.

    Parameters
    ----------
    img_halfres  : np.ndarray (H, W, 3) or (H, W)  half-res image
    tiles        : list of tile dicts from make_tiles()
    cable_thresh : int  pixel brightness threshold (0-255); pixels above this
                   are counted as cable pixels.  Default 80.

    Returns
    -------
    dict  (row, col) -> density  (float 0.0 – 1.0)
    """
    if img_halfres.ndim == 3:
        gray = cv2.cvtColor(img_halfres, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_halfres

    densities = {}
    for t in tiles:
        r0, r1, c0, c1 = t['r0'], t['r1'], t['c0'], t['c1']
        region     = gray[r0:r1, c0:c1]
        total      = (r1 - r0) * (c1 - c0)
        cable_px   = int(np.sum(region > cable_thresh))
        densities[(t['row'], t['col'])] = cable_px / total if total > 0 else 0.0
    return densities


# ─────────────────────────────────────────────────────────────────────────────
# Density-guided decluttering loop
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_new_image(watch_dir, known_files, poll_secs=3.0):
    """
    Block until a new image file appears in watch_dir.  Returns its full path.
    """
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    print(f"  Watching {watch_dir} for a new image…  (Ctrl-C to abort)")
    while True:
        current = {f for f in os.listdir(watch_dir)
                   if os.path.splitext(f)[1].lower() in exts}
        new = current - known_files
        if new:
            newest = max(new, key=lambda f:
                         os.path.getmtime(os.path.join(watch_dir, f)))
            path = os.path.join(watch_dir, newest)
            known_files.update(current)
            print(f"  New image: {path}")
            return path
        time.sleep(poll_secs)


def run_density_guided_loop(initial_image_path, output_dir, num_endpoints,
                             grid_rows=2, grid_cols=2,
                             overlap_frac=0.25, path_len_per_tile=120,
                             density_threshold=0.15, cable_thresh=80,
                             watch_dir=None, poll_secs=3.0,
                             max_iterations=50, viz=False):
    """
    Density-guided decluttering loop.

    1. Runs the full segmented tracing pipeline on the initial image.
    2. Computes per-tile cable density.
    3. Puts the orange FOCUS frame on the highest-density tile.
    4. Waits for a new image (either from watch_dir or interactive input).
    5. Recomputes densities, moves the FOCUS frame if needed.
    6. Repeats until all tiles are below density_threshold.

    The idea is that the operator (or the robot pipeline) works on the
    focused tile and then supplies a new image.  This loop tracks progress
    and always directs attention to wherever the cable tangle is densest.

    Parameters
    ----------
    initial_image_path : str
    output_dir         : str
    num_endpoints      : int
    grid_rows/cols     : int   tile grid dimensions
    overlap_frac       : float tile overlap fraction
    path_len_per_tile  : int   max tracer steps per tile
    density_threshold  : float tiles above this are considered unresolved
                               (0-1; default 0.15 = 15% cable pixel density)
    cable_thresh       : int   brightness cutoff for cable pixels (0-255)
    watch_dir          : str or None
                               if set, automatically picks up new images
                               placed in this directory; otherwise prompts
                               interactively
    poll_secs          : float how often to check watch_dir (seconds)
    max_iterations     : int   safety cap on loop iterations
    viz                : bool  open live cv2 window
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── initial run ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Density-guided loop  –  initial segmented trace")
    print("=" * 60)
    result = run_segmented_pipeline(
        initial_image_path, output_dir, num_endpoints,
        grid_rows=grid_rows, grid_cols=grid_cols,
        overlap_frac=overlap_frac, path_len_per_tile=path_len_per_tile,
        viz=False)

    # Load half-res image for density computation and display.
    bgr  = cv2.imread(initial_image_path)
    img  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_down = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    H2, W2 = img_down.shape[:2]
    tiles   = make_tiles(H2, W2, grid_rows, grid_cols, overlap_frac)

    densities = compute_tile_cable_density(img_down, tiles, cable_thresh)

    # Build the persistent visualizer (reused across all iterations).
    vis = ProgressVisualizer(img_down, tiles, result['endpoints'],
                              output_dir, show_window=viz)
    vis.update_densities(densities)

    # Track known files in watch_dir (to detect *new* arrivals only).
    known_files: set = set()
    if watch_dir and os.path.isdir(watch_dir):
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        known_files = {f for f in os.listdir(watch_dir)
                       if os.path.splitext(f)[1].lower() in exts}

    # ── main loop ─────────────────────────────────────────────────────────
    for iteration in range(max_iterations):
        # Tiles still above threshold, sorted worst-first.
        above = sorted(
            [(k, v) for k, v in densities.items() if v >= density_threshold],
            key=lambda x: x[1], reverse=True)

        if not above:
            print("\n✓  All tiles below threshold – decluttering complete!")
            break

        focus_key, focus_density = above[0]
        vis.set_attention_tile(focus_key)

        print(f"\n── Iteration {iteration + 1} ──────────────────────────────")
        print(f"   Focus tile  : {focus_key}  (density {focus_density:.1%})")
        print(f"   Tiles above threshold ({density_threshold:.0%}): "
              f"{len(above)} / {len(tiles)}")
        for k, v in above:
            marker = " ◀ FOCUS" if k == focus_key else ""
            print(f"     tile {k}  {v:.1%}{marker}")

        # Save a standalone density-map image for this iteration.
        dm_path = os.path.join(output_dir,
                                f'density_map_iter{iteration + 1:02d}.png')
        _save_density_map(img_down, tiles, densities, focus_key,
                          density_threshold, dm_path)

        # ── wait for next image ──────────────────────────────────────────
        print(f"\n   Run the decluttering pipeline on tile {focus_key}, "
              f"then provide a new image.")
        if watch_dir and os.path.isdir(watch_dir):
            new_path = _wait_for_new_image(watch_dir, known_files, poll_secs)
        else:
            new_path = input(
                "   Path to new image (Enter = re-use same image): ").strip()
            if not new_path:
                new_path = initial_image_path

        if not os.path.exists(new_path):
            print(f"   WARNING: {new_path} not found – retrying same image")
            new_path = initial_image_path

        # ── recompute densities from new image ───────────────────────────
        new_bgr  = cv2.imread(new_path)
        new_img  = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB)
        new_down = cv2.resize(new_img, (new_img.shape[1]//2, new_img.shape[0]//2))

        densities = compute_tile_cable_density(new_down, tiles, cable_thresh)
        vis.update_base_image(new_down)
        vis.update_densities(densities)

        print(f"   After update – focus tile density: "
              f"{densities.get(focus_key, 0):.1%}  "
              f"(was {focus_density:.1%})")

    else:
        print(f"\n  Reached max iterations ({max_iterations}). "
              f"Tiles still above threshold: "
              f"{sum(1 for v in densities.values() if v >= density_threshold)}")

    # Final summary image.
    _save_density_map(img_down, tiles, densities, None,
                      density_threshold,
                      os.path.join(output_dir, 'density_map_final.png'))
    vis.set_attention_tile(None)
    vis.close()
    print(f"\nOutput saved to: {output_dir}")


def _save_density_map(img_halfres, tiles, densities, focus_key,
                       density_threshold, path):
    """
    Save a standalone density-map image: coloured tile overlays + labels,
    with the focus tile highlighted and a threshold legend.
    """
    canvas = img_halfres.copy()
    overlay = canvas.copy()

    for t in tiles:
        key = (t['row'], t['col'])
        d   = densities.get(key, 0.0)
        cv2.rectangle(overlay,
                      (t['c0'], t['r0']), (t['c1'], t['r1']),
                      _density_color(d), -1)
        cx = (t['c0'] + t['c1']) // 2 - 20
        cy = (t['r0'] + t['r1']) // 2
        resolved = "✓" if d < density_threshold else "✗"
        label = f"{d:.0%} {resolved}"
        cv2.putText(overlay, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(overlay, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    canvas = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)

    # Tile borders
    for t in tiles:
        cv2.rectangle(canvas,
                      (t['c0'], t['r0']), (t['c1'], t['r1']),
                      (180, 180, 180), 1)

    # Focus frame
    if focus_key is not None:
        t = next((x for x in tiles
                  if (x['row'], x['col']) == focus_key), None)
        if t:
            r0, r1, c0, c1 = t['r0'], t['r1'], t['c0'], t['c1']
            cv2.rectangle(canvas, (c0, r0), (c1, r1), (255, 140, 0), 4)
            cv2.rectangle(canvas, (c0+5, r0+5), (c1-5, r1-5), (255,255,0), 2)
            cs = 18
            for (cy, cx), (dy, dx) in [
                ((r0, c0), ( 1,  1)), ((r0, c1), ( 1, -1)),
                ((r1, c0), (-1,  1)), ((r1, c1), (-1, -1)),
            ]:
                cv2.line(canvas, (cx, cy), (cx + dx*cs, cy), (255,255,0), 3)
                cv2.line(canvas, (cx, cy), (cx, cy + dy*cs), (255,255,0), 3)
            d = densities.get(focus_key, 0.0)
            cv2.putText(canvas, f"FOCUS  {d:.0%}",
                        (c0 + 6, r0 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, f"FOCUS  {d:.0%}",
                        (c0 + 6, r0 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Legend
    lx, ly = 8, 20
    cv2.putText(canvas, f"threshold: {density_threshold:.0%}",
                (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
    cv2.putText(canvas, f"threshold: {density_threshold:.0%}",
                (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Segmented cable tracing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python segmented_tracer.py --image input.png --tier 2
  python segmented_tracer.py --image input.png --tier 4 --grid_rows 3 --grid_cols 3
  python segmented_tracer.py --image input.png --tier 2 --overlap 0.3 --path_len 80 --viz
        """)
    parser.add_argument('--image',      required=True,
                        help='Path to input image')
    parser.add_argument('--tier',       type=int, required=True,
                        help='1-8 – determines expected endpoint count')
    parser.add_argument('--grid_rows',  type=int, default=2,
                        help='Tile-grid rows (default 2)')
    parser.add_argument('--grid_cols',  type=int, default=2,
                        help='Tile-grid cols (default 2)')
    parser.add_argument('--overlap',    type=float, default=0.25,
                        help='Fractional tile overlap, 0–1 (default 0.25)')
    parser.add_argument('--path_len',   type=int, default=120,
                        help='Max tracer steps per tile (default 120)')
    parser.add_argument('--output_dir',  default=None,
                        help='Output directory (default: auto timestamp)')
    parser.add_argument('--viz',         action='store_true',
                        help='Open a live cv2 window (requires display)')
    # ── density-guided loop options ──────────────────────────────────────
    parser.add_argument('--density_loop', action='store_true',
                        help='Run the density-guided decluttering loop '
                             'instead of a single-shot trace')
    parser.add_argument('--threshold',   type=float, default=0.15,
                        help='Cable density threshold; tiles above this are '
                             'considered unresolved (default 0.15 = 15%%)')
    parser.add_argument('--cable_thresh', type=int, default=80,
                        help='Pixel brightness cutoff for cable detection '
                             '(0-255, default 80)')
    parser.add_argument('--watch_dir',   default=None,
                        help='Directory to watch for new images between '
                             'iterations (default: interactive prompt)')
    parser.add_argument('--poll_secs',   type=float, default=3.0,
                        help='How often to poll watch_dir (seconds, default 3)')
    parser.add_argument('--max_iter',    type=int, default=50,
                        help='Safety cap on density-loop iterations (default 50)')

    args = parser.parse_args()

    if args.tier not in TIER_TO_ENDPOINTS:
        print(f"Error: tier must be 1-8, got {args.tier}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    num_endpoints = TIER_TO_ENDPOINTS[args.tier]

    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(SAVE_DIR, "segmented", ts)
    os.makedirs(output_dir, exist_ok=True)

    if args.density_loop:
        run_density_guided_loop(
            args.image, output_dir, num_endpoints,
            grid_rows=args.grid_rows, grid_cols=args.grid_cols,
            overlap_frac=args.overlap, path_len_per_tile=args.path_len,
            density_threshold=args.threshold, cable_thresh=args.cable_thresh,
            watch_dir=args.watch_dir, poll_secs=args.poll_secs,
            max_iterations=args.max_iter, viz=args.viz)
    else:
        run_segmented_pipeline(
            args.image, output_dir, num_endpoints,
            grid_rows=args.grid_rows, grid_cols=args.grid_cols,
            overlap_frac=args.overlap, path_len_per_tile=args.path_len,
            viz=args.viz)


if __name__ == '__main__':
    main()
