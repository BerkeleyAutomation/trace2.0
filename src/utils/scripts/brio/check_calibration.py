#!/usr/bin/env python3
"""Visualize BRIO calibration error for the BWW setup without robot motion.

Captures a live image, detects the 8x6 chessboard, projects expected corner
positions using the current brio_to_world_bww.tf calibration, and overlays
both on the image to reveal the magnitude and direction of any error.

Usage:
    python check_calibration.py [--device 1] [--square-size 0.03]

Place chessboard at the calibration position, run the script, press Enter
to capture.  Green dots = detected corners, red dots = expected (projected)
corners.  If calibration is correct they should overlap (< 5 px RMS).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from autolab_core import RigidTransform

SCRIPT_DIR = Path(__file__).resolve().parent

# Must match calibrate_brio_bww.py
INNER_CORNERS_X = 8
INNER_CORNERS_Y = 6


# ---------------------------------------------------------------------------
# Helpers (inlined from calibrate_brio_bww.py to avoid cross-package imports)
# ---------------------------------------------------------------------------

def get_cropped_undistorted_intrinsics(cam) -> np.ndarray:
    """Return 3x3 K for the undistorted, ROI-cropped image returned by cam.read()."""
    x, y, _, _ = cam.roi
    K = cam.newcameramtx.copy()
    K[0, 2] -= x
    K[1, 2] -= y
    return K


def make_object_points(square_size_m: float) -> np.ndarray:
    """Return (N,3) object points centered at the chessboard origin."""
    grid = np.mgrid[0:INNER_CORNERS_X, 0:INNER_CORNERS_Y].T.reshape(-1, 2).astype(np.float32)
    grid -= np.array(
        [(INNER_CORNERS_X - 1) / 2.0, (INNER_CORNERS_Y - 1) / 2.0],
        dtype=np.float32,
    )
    objp = np.zeros((INNER_CORNERS_X * INNER_CORNERS_Y, 3), dtype=np.float32)
    objp[:, :2] = grid * square_size_m
    return objp


def detect_corners(image_rgb: np.ndarray) -> np.ndarray:
    """Find sub-pixel chessboard corners.  Returns (N,1,2) float32 array."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (INNER_CORNERS_X, INNER_CORNERS_Y), flags)
    if not found:
        raise RuntimeError(
            f"Could not find a {INNER_CORNERS_X}x{INNER_CORNERS_Y} inner-corner chessboard. "
            "Make sure the board is fully visible and well-lit."
        )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def project_world_to_pixels(
    world_pts_h: np.ndarray,
    T_brio_base: RigidTransform,
    K: np.ndarray,
) -> np.ndarray:
    """Project Nx4 homogeneous world (base_link) points to Nx2 pixel coords."""
    T_base_brio = T_brio_base.inverse()
    cam_pts = (T_base_brio.matrix @ world_pts_h.T).T  # Nx4
    proj = (K @ cam_pts[:, :3].T).T                   # Nx3
    pixels = proj[:, :2] / proj[:, 2:3]
    return pixels.astype(np.float32)


def draw_chessboard_guide(
    image_rgb: np.ndarray,
    T_cb_base: RigidTransform,
    T_brio_base: RigidTransform,
    K: np.ndarray,
    square_size_m: float,
    scale: float,
) -> np.ndarray:
    """Draw projected chessboard boundary, center, and X/Y axes on a scaled copy."""
    # World-space points to project
    half_x = (INNER_CORNERS_X - 1) / 2.0 * square_size_m
    half_y = (INNER_CORNERS_Y - 1) / 2.0 * square_size_m
    axis_len = 3 * square_size_m  # length of drawn axis arrow in meters

    # Origin + axis tips in cb frame (homogeneous)
    pts_cb = np.array([
        [0,       0,       0, 1],  # origin / center
        [axis_len, 0,       0, 1],  # +X tip
        [0,       axis_len, 0, 1],  # +Y tip
        # Four corners of the inner-corner bounding box
        [ half_x,  half_y, 0, 1],
        [-half_x,  half_y, 0, 1],
        [-half_x, -half_y, 0, 1],
        [ half_x, -half_y, 0, 1],
    ], dtype=np.float64)

    world_pts = (T_cb_base.matrix @ pts_cb.T).T  # Nx4
    pix = project_world_to_pixels(world_pts, T_brio_base, K)  # Nx2

    h, w = image_rgb.shape[:2]
    vis = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    def sp(i):  # scaled pixel as int tuple
        return (int(pix[i, 0] * scale), int(pix[i, 1] * scale))

    origin = sp(0)
    x_tip  = sp(1)
    y_tip  = sp(2)
    box    = [sp(3), sp(4), sp(5), sp(6)]

    lw = max(2, int(3 / scale * scale))  # line width

    # Bounding box of expected corners (white)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(vis, box[a], box[b], (255, 255, 255), lw)

    # X axis = red, Y axis = green
    cv2.arrowedLine(vis, origin, x_tip, (0, 0, 255),   lw + 1, tipLength=0.2)
    cv2.arrowedLine(vis, origin, y_tip, (0, 255, 0),   lw + 1, tipLength=0.2)
    cv2.circle(vis, origin, max(6, lw * 3), (0, 255, 255), -1)

    # Labels
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    cv2.putText(vis, "X (base_link)", x_tip, font, fs, (0, 0, 255),   ft)
    cv2.putText(vis, "Y (base_link)", y_tip, font, fs, (0, 255, 0),   ft)
    cv2.putText(vis, "PLACE BOARD CENTER HERE", origin, font, fs,
                (0, 255, 255), ft)

    # HUD
    cv2.putText(vis,
                "PREVIEW: align board to white box  |  Press ENTER in terminal to capture",
                (10, 30), font, 0.65, (200, 200, 0), 2)

    return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check BRIO-to-base calibration visually (no robot motion required)"
    )
    parser.add_argument("--device", type=int, default=1, help="BRIO /dev/video index")
    parser.add_argument(
        "--square-size", type=float, default=0.03, help="Chessboard square size in meters"
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=SCRIPT_DIR / "brio_to_world_bww.tf",
        help="Camera-to-base_link transform (.tf)",
    )
    parser.add_argument(
        "--chessboard-tf",
        type=Path,
        default=SCRIPT_DIR / "bww_chessboard.tf",
        help="Chessboard-to-base_link transform (.tf)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.25,
        help="Display scale factor (default 0.25 for 3840x2160 camera)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    # Import BRIOSensor from the same directory without requiring a package install
    sys.path.insert(0, str(SCRIPT_DIR))
    from brio_sensor import BRIOSensor  # noqa: PLC0415

    print(f"Loading calibration :  {args.calib}")
    print(f"Loading chessboard TF: {args.chessboard_tf}")
    T_brio_base = RigidTransform.load(str(args.calib))
    T_cb_base   = RigidTransform.load(str(args.chessboard_tf))

    print(f"\nInitializing BRIO device {args.device} ...")
    cam = BRIOSensor(args.device)
    cam.start()

    try:
        print(
            "\nPlace the chessboard at the fixed BWW calibration position, then"
            " press Enter to capture."
        )
        input()

        # Flush stale frames from the buffer
        for _ in range(5):
            image = cam.read()

        print("Detecting chessboard corners ...")
        corners_detected = detect_corners(image)          # (N,1,2)
        N = corners_detected.shape[0]
        pixels_detected = corners_detected.reshape(N, 2)  # (N,2)

        K = get_cropped_undistorted_intrinsics(cam)
        objp = make_object_points(args.square_size)       # (N,3)

        # Transform object points: cb frame → base_link frame
        objp_h = np.hstack([objp, np.ones((N, 1), dtype=np.float32)])  # Nx4
        world_pts_h = (T_cb_base.matrix @ objp_h.T).T                   # Nx4

        # Project expected corners using the current calibration
        pixels_expected = project_world_to_pixels(world_pts_h, T_brio_base, K)  # Nx2

        # -------------------------------------------------------------------
        # Error metrics
        # -------------------------------------------------------------------
        errors = pixels_detected - pixels_expected          # Nx2
        error_norms = np.linalg.norm(errors, axis=1)        # N
        rms_px = float(np.sqrt(np.mean(error_norms ** 2)))

        # Approximate depth to estimate world-space error
        T_base_brio = T_brio_base.inverse()
        cam_pts = (T_base_brio.matrix @ world_pts_h.T).T
        approx_depth_m = float(np.mean(cam_pts[:, 2]))
        fx = float(K[0, 0])
        world_error_m = rms_px / fx * approx_depth_m

        # -------------------------------------------------------------------
        # Per-corner table
        # -------------------------------------------------------------------
        print(f"\n{'#':>4}  {'det_x':>7} {'det_y':>7}  {'exp_x':>7} {'exp_y':>7}  {'err_px':>7}  {'dx':>7} {'dy':>7}")
        print("-" * 70)
        for i in range(N):
            dx_px, dy_px = errors[i]
            print(
                f"{i:>4}  "
                f"{pixels_detected[i, 0]:>7.1f} {pixels_detected[i, 1]:>7.1f}  "
                f"{pixels_expected[i, 0]:>7.1f} {pixels_expected[i, 1]:>7.1f}  "
                f"{error_norms[i]:>7.2f}  "
                f"{dx_px:>+7.1f} {dy_px:>+7.1f}"
            )

        print(f"\nRMS pixel error:        {rms_px:.2f} px")
        print(f"Approx camera depth:    {approx_depth_m:.3f} m")
        print(f"Est. XY world error:    {world_error_m * 1000:.1f} mm  "
              f"({world_error_m * 39.37:.2f} in)")

        mean_err_vec_m = np.mean(errors, axis=0) / fx * approx_depth_m
        print(f"Mean world offset:      dx={mean_err_vec_m[0]*1000:+.1f} mm  "
              f"dy={mean_err_vec_m[1]*1000:+.1f} mm")

        if rms_px < 5:
            print("\n[GOOD]  Calibration looks correct (< 5 px RMS)")
        else:
            print(f"\n[WARN]  Calibration is off by ~{rms_px:.0f} px RMS — "
                  f"see overlay for direction")

        # -------------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------------
        vis = image.copy()

        # Draw at full resolution; dot/line sizes chosen to be visible after downscale
        dot_radius   = max(8, int(15 / args.scale))
        line_width   = max(2, int(4  / args.scale))
        dot_thickness = -1  # filled

        for i in range(N):
            det = tuple(int(v) for v in pixels_detected[i])
            exp = tuple(int(v) for v in pixels_expected[i])
            cv2.line(vis, det, exp, (0, 255, 255), line_width)   # cyan error vector
            cv2.circle(vis, det, dot_radius, (0, 255, 0), dot_thickness)   # green: detected
            cv2.circle(vis, exp, dot_radius, (0, 0, 255), dot_thickness)   # red:   expected

        h, w = vis.shape[:2]
        small = cv2.resize(vis, (int(w * args.scale), int(h * args.scale)))
        small_bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

        window_title = (
            f"RMS={rms_px:.1f}px  world~{world_error_m*1000:.0f}mm  |  "
            "green=detected  red=expected (projected)  cyan=error"
        )
        cv2.imshow(window_title, small_bgr)
        print("\nPress any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        cam.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
