"""
Live analysis dashboard for the decluttering pipeline.

Monitors the pipeline output directory and displays all analysis images
(raw image, object masks, endpoint detection, cable traces, etc.)
compiled into a single auto-updating window.

Run this in parallel with the robot or vision pipeline:
    python analysis_dashboard.py --output_dir <pipeline_output_dir> [--refresh 2.0]

Or point it at an existing completed run:
    python analysis_dashboard.py --output_dir <pipeline_output_dir> --once
    
    
    conda activate trace && python /home/justinyu/multicable-decluttering/decluttering/src/decluttering_pipeline_robot_parallel.py --tier 3 --output_dir /home/justinyu/multicable-decluttering/decluttering/src/IROS26_video --dashboard
"""

import argparse
import glob
import math
import os
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Recognized image filenames produced by the pipeline (in display order)
PANEL_NAMES = [
    ("raw_image.png", "Raw Image"),
    # ("endpoints.png", "Endpoints"),
    ("object_masks.png", "Object Masks"),
    ("combined_traces.png", "Combined Traces"),
    ("divergence_point.png", "Divergence Point"),
]
# Individual trace images are matched by pattern: trace_*.png
TRACE_PATTERN = "trace_*.png"


def load_image(path):
    """Load an image as RGB numpy array, or return None."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_latest_iteration(output_dir):
    """Find all iteration subdirectories, sorted by index."""
    iter_dirs = sorted(
        glob.glob(os.path.join(output_dir, "iter_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1])
    )
    return iter_dirs


def collect_panels(iter_dir):
    """Collect all displayable images from an iteration directory.

    Returns list of (image_array, title) tuples.
    """
    panels = []

    # Fixed-name panels
    for filename, title in PANEL_NAMES:
        img = load_image(os.path.join(iter_dir, filename))
        if img is not None:
            panels.append((img, title))

    # Individual trace panels
    trace_files = sorted(glob.glob(os.path.join(iter_dir, TRACE_PATTERN)))
    # for tf in trace_files:
    #     idx = os.path.splitext(os.path.basename(tf))[0]  # e.g. "trace_0"
    #     img = load_image(tf)
    #     if img is not None:
    #         panels.append((img, f"Trace {idx.split('_')[-1]}"))

    return panels


def read_metrics(output_dir):
    """Read the metrics log file if it exists."""
    metrics_path = os.path.join(output_dir, "metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return f.read()
    return None


def read_status(output_dir):
    """Read the current pipeline status if it exists."""
    status_path = os.path.join(output_dir, "status.txt")
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            return f.read().strip()
    return None


def build_dashboard(fig, output_dir):
    """Build/rebuild the full dashboard figure. Returns True if any content was shown."""
    fig.clear()

    iter_dirs = find_latest_iteration(output_dir)
    if not iter_dirs:
        # Maybe a flat directory (vision pipeline output, no iter_ subdirs)
        iter_dirs = [output_dir]

    # Collect panels from all iterations
    all_iter_panels = []
    for d in iter_dirs:
        panels = collect_panels(d)
        if panels:
            all_iter_panels.append((os.path.basename(d), panels))

    if not all_iter_panels:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Waiting for pipeline output...",
                ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')
        return False

    # Most recent iteration first
    all_iter_panels = list(reversed(all_iter_panels))

    RECENT_COLS = 4  # panels per row for the most recent iteration
    recent_name, recent_panels = all_iter_panels[0]
    older = all_iter_panels[1:]

    # Separate divergence point panel so it can be displayed larger
    div_idx = next((i for i, (_, t) in enumerate(recent_panels) if t == "Divergence Point"), None)
    if False and div_idx is not None:
        other_panels = [p for i, p in enumerate(recent_panels) if i != div_idx]
        div_panel = recent_panels[div_idx]
    else:
        other_panels = recent_panels
        div_panel = None

    n_other = len(other_panels)
    other_rows = math.ceil(n_other / RECENT_COLS) if n_other > 0 else 0
    recent_rows = max(other_rows + (1 if div_panel is not None else 0), 1)
    # Divergence row gets 2x height relative to other rows
    recent_h_ratios = [1] * other_rows + ([2] if div_panel is not None else []) or [1]

    DIV_SPAN = 2  # divergence point spans this many columns (centered)
    div_col_start = (RECENT_COLS - DIV_SPAN) // 2

    if older:
        n_older_rows = len(older)
        max_older_cols = max(len(p) for _, p in older)
        # Recent rows get 2x height of older rows
        outer_height_ratios = [2] * recent_rows + [1] * n_older_rows
        total_rows = recent_rows + n_older_rows
        gs_outer = gridspec.GridSpec(total_rows, 1, figure=fig,
                                     height_ratios=outer_height_ratios, hspace=0.15)
        gs_recent = gridspec.GridSpecFromSubplotSpec(
            recent_rows, RECENT_COLS,
            subplot_spec=gs_outer[:recent_rows, 0],
            height_ratios=recent_h_ratios,
            hspace=0.08, wspace=0.02)
        gs_older = gridspec.GridSpecFromSubplotSpec(
            n_older_rows, max_older_cols,
            subplot_spec=gs_outer[recent_rows:, 0],
            hspace=0.08, wspace=0.02)
        for row, (iter_name, panels) in enumerate(older):
            for col, (img, title) in enumerate(panels):
                ax = fig.add_subplot(gs_older[row, col])
                ax.imshow(img)
                ax.set_title(f"{iter_name}: {title}", fontsize=7)
                ax.axis('off')
    else:
        gs_recent = gridspec.GridSpec(recent_rows, RECENT_COLS, figure=fig,
                                      height_ratios=recent_h_ratios,
                                      hspace=0.08, wspace=0.02)

    for i, (img, title) in enumerate(other_panels):
        ax = fig.add_subplot(gs_recent[i // RECENT_COLS, i % RECENT_COLS])
        ax.imshow(img)
        ax.set_title(f"{recent_name}: {title}", fontsize=9)
        ax.axis('off')

    if div_panel is not None:
        ax = fig.add_subplot(gs_recent[other_rows, div_col_start:div_col_start + DIV_SPAN])
        img, title = div_panel
        ax.imshow(img)
        ax.set_title(f"{recent_name}: {title}", fontsize=11, fontweight='bold')
        ax.axis('off')

    # Show metrics as suptitle (truncated)
    metrics = read_metrics(output_dir)
    if metrics:
        lines = metrics.strip().split('\n')
        summary = '\n'.join(lines[:3])  # first few lines
        fig.suptitle(summary, fontsize=10, family='monospace', y=0.995)

    # Show live status bar at bottom
    status = read_status(output_dir)
    if status:
        fig.text(0.5, 0.002, f"STATUS:  {status}",
                 ha='center', va='bottom', fontsize=12, fontweight='bold',
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='steelblue', alpha=0.9))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
    return True


def launch_dashboard(output_dir, refresh=2.0):
    """Launch the dashboard as a background subprocess. Returns the Popen object."""
    script_path = os.path.abspath(__file__)
    proc = subprocess.Popen(
        [sys.executable, script_path, "--output_dir", output_dir, "--refresh", str(refresh)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Dashboard launched (PID {proc.pid}) monitoring: {output_dir}")
    return proc


def run_dashboard(output_dir, refresh_interval, once=False):
    """Main dashboard loop."""
    plt.ion()
    fig = plt.figure("Decluttering Dashboard", figsize=(52, 36))

    try:
        while True:
            build_dashboard(fig, output_dir)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if once:
                plt.ioff()
                plt.show()
                return

            plt.pause(refresh_interval)

    except KeyboardInterrupt:
        print("\nDashboard closed.")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Live analysis dashboard for decluttering pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch a live pipeline run:
  python analysis_dashboard.py --output_dir /path/to/pipeline/output

  # View a completed run (no auto-refresh):
  python analysis_dashboard.py --output_dir /path/to/pipeline/output --once

  # Custom refresh rate:
  python analysis_dashboard.py --output_dir /path/to/pipeline/output --refresh 5.0
        """
    )
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Pipeline output directory to monitor')
    parser.add_argument('--refresh', type=float, default=2.0,
                        help='Refresh interval in seconds (default: 2.0)')
    parser.add_argument('--once', action='store_true',
                        help='Show dashboard once and block (no auto-refresh)')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"Error: Directory not found: {args.output_dir}")
        print("Start the pipeline first, or provide an existing output directory.")
        sys.exit(1)

    print(f"Monitoring: {args.output_dir}")
    print(f"Refresh interval: {args.refresh}s")
    print("Press Ctrl+C to close.\n")

    run_dashboard(args.output_dir, args.refresh, args.once)


if __name__ == '__main__':
    main()
