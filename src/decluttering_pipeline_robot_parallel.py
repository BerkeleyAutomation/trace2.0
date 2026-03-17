"""
Robot decluttering pipeline with parallelized vision steps.

Same as decluttering_pipeline_robot.py but runs:
  1. Detic object detection + endpoint detection in parallel
  2. Per-endpoint cable tracing in parallel

Usage:
    python decluttering_pipeline_robot_parallel.py --tier <1-4> [--output_dir <dir>] [--viz]


TODO: 

SETUP FOR PYROKI 

What you need to do (on the robot controller):                                                                                          
                                                                                                                                                        
1. Start the RAPID EGM program on the FlexPendant — the EGM session must be running before YuMiROSInterface can stream poses.                          
2. Verify the UDP target IP in the RAPID code matches your machine's current IP. The IP is set in the ABB controller's EGM configuration (in the RAPID 
EGMSetupUC or similar call), not in any Python file here.   

"""

### 2/20 Successful ran on trace_copy conda environment
### If you get a detectron error, run cd /home/justinyu/multicable-decluttering/decluttering/data/detectron2_repo && CUDA_HOME=/usr/local/cuda-11.8 python setup.py build_ext --inplace

# ---- HACK: Fix detectron issue

import subprocess
command = "cd /home/justinyu/multicable-decluttering/decluttering/data/detectron2_repo && CUDA_HOME=/usr/local/cuda-11.8 python setup.py build_ext --inplace"
result = subprocess.run(command, shell = True)

# -----

# Tier to endpoints mapping

TIER_TO_ENDPOINTS = {1: 4, 2: 4, 3: 6, 4: 8, 5: 10, 6:12, 7:14, 8:16}

import argparse
import copy
import cv2
import json
import numpy as np
import math
import time
import os
import sys
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.append('/home/justinyu/multicable-decluttering/')

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; avoids tkinter conflicts with ThreadPoolExecutor
import matplotlib.pyplot as plt

# Robot interface
import threading
from yumi_realtime.yumi_realtime.controller import YuMiROSInterface

# Camera
from utils.scripts.brio.brio_sensor import BRIOSensor

# Vision (reuse from vision script)
from main_vision import (
    dilate_masks, get_detic_masks, predict_pts,
    save_object_masks_image, save_trace_images
)

# Vision models
from utils.detic_dataloader import DeticDataloader
from utils.tracer_dataloader import TracerDataloader
from utils.endpoint_detector_dataloader import EndpointDataloader

# Vision utilities
from divergencecopy import get_all_div_points
from main_vision import get_trace_list_colored_parallel as get_trace_list_colored
from knot_regions import find_open_knot_regions, classify_point_in_knot
from run_constants import *

# Per-endpoint trace helpers
from utils.tracer.tusk_pipeline.tracer import TraceEnd
from masker import get_mask

# Robot motions
from motion_pyroki import (
    goto_gripper_px,
    rotate_gripper,
    perform_push_through,
    perform_point_grasp_and_bin_drop_pipeline,
    bimanual_grasp_and_bin_drop_pipeline,
    get_push_coords
)

# Hardcoded constants
NUM_MOVES = 10
BIMANUAL_DECLUTTER = True  # Enable bimanual mode (matches original pipeline)
ENDPOINT_EXCLUSION_RADIUS = 200  # pixels in full-res; masks within this radius of an endpoint are not picked up (e.g. hubs)


# ---------------------------------------------------------------------------
# Helper functions (unchanged from decluttering_pipeline_robot.py)
# ---------------------------------------------------------------------------

def get_oriented_bounding_box(binary_mask):
    """
    Fit the tightest oriented bounding box around a 2D binary mask.

    Parameters:
        binary_mask: A 2D numpy array representing the binary mask image.

    Returns:
        tuple: (rect, angle, largest_contour)
    """
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)

    angle = rect[2]
    if rect[1][0] > rect[1][1]:
        angle = angle + 90
    if np.cos(np.radians(angle)) > 0:
        angle = angle + 180

    return rect, angle, largest_contour


def check_valid_bimanual_grasp(pair_indices, out, viz_mask_pair=False):
    """Check if a pair of masks can be grasped bimanually."""
    mask_pair = out['masks'][list(pair_indices)]
    img_width = mask_pair.shape[3]

    if viz_mask_pair:
        plt.imshow(mask_pair[0].squeeze().cpu().numpy())
        plt.imshow(mask_pair[1].squeeze().cpu().numpy(), alpha=0.5)
        plt.show()

    obb1, _, _ = get_oriented_bounding_box(mask_pair[0].squeeze().cpu().numpy())
    obb2, _, _ = get_oriented_bounding_box(mask_pair[1].squeeze().cpu().numpy())

    arm_config_coverage = 0.55

    obb1_left_graspable = obb1[0][0] < img_width * arm_config_coverage
    obb1_right_graspable = obb1[0][0] > img_width * (1 - arm_config_coverage)
    obb2_left_graspable = obb2[0][0] < img_width * arm_config_coverage
    obb2_right_graspable = obb2[0][0] > img_width * (1 - arm_config_coverage)

    distance = np.linalg.norm(np.array(obb1[0]) - np.array(obb2[0]))
    print(f"Distance: {distance}")

    if distance < 600:
        print("Distance too close, moving to next pair")
        return False, False

    if obb1_left_graspable and obb2_right_graspable:
        return True, False
    if obb1_right_graspable and obb2_left_graspable:
        return True, True

    return False, False


def bimanual_filter_grasps(mask_num, out):
    """Filter grasps for bimanual mode."""
    assert isinstance(mask_num, np.ndarray), f"mask_num should be a numpy array but is instead: {type(mask_num)}"

    if mask_num.all() == -1:
        return -1

    masks = np.delete(mask_num, np.where(mask_num == -1))
    masks_set = set(masks)

    if len(masks_set) == 1:
        return [int(list(masks_set)[0])]

    for pair in itertools.combinations(masks_set, 2):
        valid, flip_lr = check_valid_bimanual_grasp(pair, out)
        if valid:
            if flip_lr:
                print(f"Flipping left and right grasp for pair {pair}")
                pair = [pair[1], pair[0]]
            return list(pair)

    if len(masks_set) > 0:
        print("No valid bimanual grasp found, returning single arm grasp")
        return [int(masks_set.pop())]
    return -1


def is_mask_near_endpoint(mask, endpoints):
    """Return True if the mask centroid is within ENDPOINT_EXCLUSION_RADIUS of any endpoint.

    mask: tensor of shape (1, H, W) in full-resolution space
    endpoints: list of (row, col) in half-resolution space
    """
    mask_np = mask[0].cpu().numpy()
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        return False
    centroid = np.array([ys.mean(), xs.mean()])
    for endpt in endpoints:
        endpt_full = np.array([endpt[0] * 2, endpt[1] * 2])  # scale to full-res
        if np.linalg.norm(centroid - endpt_full) < ENDPOINT_EXCLUSION_RADIUS:
            return True
    return False


def filter_hub_masks(mask_num, detic_out, endpoints):
    """Remove mask indices that are near cable endpoints (hubs) from mask_num."""
    if mask_num == -1:
        return -1
    filtered = [i for i in mask_num if not is_mask_near_endpoint(detic_out['masks'][i], endpoints)]
    if len(filtered) == 0:
        print("All detected masks are near endpoints (hubs) — skipping declutter.")
        return -1
    return filtered


def write_status(output_dir, message):
    """Write current pipeline status to status.txt for the dashboard to display."""
    with open(os.path.join(output_dir, "status.txt"), "w") as f:
        f.write(message)


def save_divergence_image(img, div_points, trace_list, iter_path, zoom_pad=350):
    """Save a zoomed image marking the divergence point(s) the robot will act on."""
    vis = img.copy()

    # Draw all traces faintly — points are (x, y) = (col, row)
    for t_idx, trace in enumerate(trace_list):
        color = COLORS[t_idx % len(COLORS)]
        for pt in trace:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, color, -1)

    # Draw divergence points with a prominent marker — div_points are (x, y)
    for dp in div_points:
        x, y = int(dp[0]), int(dp[1])
        cv2.drawMarker(vis, (x, y), (255, 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        cv2.circle(vis, (x, y), DENSITY_RADIUS, (255, 0, 0), 2)

    # Zoom in around the primary divergence point
    primary = div_points[0]
    x, y = int(primary[0]), int(primary[1])
    h, w = vis.shape[:2]
    r0 = max(0, y - zoom_pad)
    r1 = min(h, y + zoom_pad)
    c0 = max(0, x - zoom_pad)
    c1 = min(w, x + zoom_pad)
    zoomed = vis[r0:r1, c0:c1]

    cv2.imwrite(os.path.join(iter_path, 'divergence_point.png'),
                cv2.cvtColor(zoomed, cv2.COLOR_RGB2BGR))


def initialize_robot():
    """Initialize YuMi robot interface using yumi_realtime."""
    print("creating interface")
    interface = YuMiROSInterface()

    # run() is a blocking 150Hz control loop — start it in a background thread
    t = threading.Thread(target=interface.run, daemon=True)
    t.start()

    # Wait until joint state callbacks have fired (cartesian_pose populated)
    print("waiting for interface to be ready")
    while interface.cartesian_pose_L is None or interface.cartesian_pose_R is None:
        time.sleep(0.1)

    print("moving to pre-home")
    interface.pre_home()
    time.sleep(2.0)  # T_RETRACT

    print("moving to home")
    interface.home()
    time.sleep(4.0)  # T_HOME

    print("calibrating grippers")
    interface.calib_gripper('left')
    interface.calib_gripper('right')

    interface.call_gripper('left',  False, True)
    interface.call_gripper('right', False, True)
    time.sleep(1.0)  # T_GRIPPER

    interface.call_gripper('left',  True, True)
    interface.call_gripper('right', True, True)
    time.sleep(1.0)

    return interface


def capture_image(cam):
    """Capture image from BRIO camera (clears buffer)."""
    for _ in range(5):
        img = cam.read()
    return img


def execute_declutter(interface, out, mask_num):
    """Execute pick and place to declutter objects."""
    mask = out['masks'][mask_num]
    bin_mask = mask[0].squeeze().cpu().numpy()
    obb, angle, _ = get_oriented_bounding_box(bin_mask)
    grasp_vec = [-np.sin(np.radians(angle)), -np.cos(np.radians(angle))]
    cmd_angle = np.degrees(np.arctan2(grasp_vec[1], grasp_vec[0]))
    center = np.array(obb[0])

    if BIMANUAL_DECLUTTER and mask.shape[0] == 2:
        # Bimanual grasp
        bin_mask_2 = mask[1].squeeze().cpu().numpy()
        obb2, angle2, _ = get_oriented_bounding_box(bin_mask_2)
        grasp_vec2 = [-np.sin(np.radians(angle2)), -np.cos(np.radians(angle2))]
        cmd_angle2 = np.degrees(np.arctan2(grasp_vec2[1], grasp_vec2[0]))
        center2 = np.array(obb2[0])
        bimanual_grasp_and_bin_drop_pipeline(center, cmd_angle, center2, cmd_angle2, interface)
    else:
        # Single arm grasp
        interface.home()
        interface.open_grippers()
        perform_point_grasp_and_bin_drop_pipeline(center, cmd_angle, interface)


def execute_knot_dilation(interface, centroid):
    """Execute knot dilation maneuver."""
    arm = goto_gripper_px(centroid * 2, interface)
    interface.open_grippers()
    rotate_gripper(arm, interface)
    interface.home()
    interface.close_grippers()


def plan_push_through(div_points, img_down, trace_list, endpt_1, endpt_2, img_rgb=None, iter_path=None):
    """Plan the push-through IP maneuver — returns coords and selected div_point without moving."""
    end_coord, start_coord, vec_angle, vec_dist = get_push_coords(
        div_points, img_down, trace_list, endpt_1, endpt_2, viz=False
    )
    selected_div_point = end_coord.copy()
    end_coord *= 2
    start_coord *= 2
    vector_0 = (end_coord[0] - start_coord[0]) * VECTOR_SCALE
    vector_1 = (end_coord[1] - start_coord[1]) * VECTOR_SCALE
    new_vec0 = start_coord[0] + vector_0
    new_vec1 = start_coord[1] + vector_1
    if img_rgb is not None and iter_path is not None:
        vis = img_rgb.copy()
        # start_coord is the push START (pixel coords: [x, y])
        sx, sy = int(start_coord[0]), int(start_coord[1])
        # [new_vec0, new_vec1] is the push END
        ex, ey = int(new_vec0), int(new_vec1)
        # Draw start (green) and end (red) circles
        cv2.circle(vis, (sx, sy), 18, (0, 255, 0), -1)   # green = start
        cv2.circle(vis, (ex, ey), 18, (255, 0, 0), -1)    # red = end
        # Draw arrow from start to end (cyan)
        cv2.arrowedLine(vis, (sx, sy), (ex, ey), (0, 255, 255), 4, tipLength=0.15)
        # Labels
        cv2.putText(vis, f"START ({sx},{sy})", (sx + 20, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(vis, f"END ({ex},{ey})", (ex + 20, ey - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        save_path = os.path.join(iter_path, 'push_through_sanity.png')
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[SANITY CHECK] Saved push-through overlay to {save_path}")
        print(f"[SANITY CHECK] Start (green): ({sx}, {sy}), End (red): ({ex}, {ey})")
    return [new_vec0, new_vec1], start_coord, vec_angle, vec_dist, selected_div_point


def execute_push_through(interface, div_points, img_down, trace_list, endpt_1, endpt_2, img_rgb=None, iter_path=None):
    """Execute push-through IP maneuver."""
    push_target, start_coord, vec_angle, vec_dist, selected_div_point = plan_push_through(
        div_points, img_down, trace_list, endpt_1, endpt_2, img_rgb, iter_path
    )
    perform_push_through(push_target, start_coord, interface)
    return vec_angle, vec_dist, selected_div_point


# ---------------------------------------------------------------------------
# Parallel vision helpers
# ---------------------------------------------------------------------------

def _trace_single_endpoint(ep_idx, img_up, endpts, learned_tracer, analytic_tracer, min_condition_pts):
    """Trace a single endpoint. Thread-safe: reads shared tracer state but does not modify it."""
    img_mask = get_mask(img_up, endpts[ep_idx])
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
    start_pts, analytic_trace_end = analytic_tracer.trace(img_mask, np.array(endpts[ep_idx]), path_len=3)
    start_pts = np.array(start_pts) if start_pts is not None else np.empty((0, 2))

    filtered_endpts = copy.deepcopy(endpts)
    filtered_endpts[ep_idx] = np.array([-100, -100])

    if len(start_pts) < min_condition_pts:
        print(
            f"Analytic tracer returned {len(start_pts)} start point(s) for trace {ep_idx}, "
            f"requires >= {min_condition_pts}. Falling back to short analytic trace."
        )
        trace_path = np.flip(start_pts, axis=1) if start_pts.size else np.array([endpts[ep_idx][::-1]])
        trace_end = analytic_trace_end
        t_endpoint = None
        densities = np.array([])
        mask_num = -1
    else:
        output = learned_tracer.trace(img_mask, start_pts, filtered_endpts, path_len=250, use_vit=False)
        trace_path = np.flip(output["trace"], axis=1)
        trace_end = output["trace_end"]
        t_endpoint = output["t_endpoint"]
        densities = output["densities"]
        mask_num = output["mask_num"]

    # Build endpt_table entry
    if trace_end == TraceEnd.ENDPOINT and t_endpoint is not None:
        equality = np.all(filtered_endpts == t_endpoint, axis=1)
        match_idx = np.nonzero(equality)[0]
        endpt_entry = [ep_idx, int(match_idx[0]) if len(match_idx) > 0 else -1]
    else:
        if trace_end == TraceEnd.EDGE:
            print(f"\nHit edge of trace, appending [{ep_idx}, -1]")
        endpt_entry = [ep_idx, -1]

    # Normalize densities
    densities = np.array(densities)
    if densities.size == 0:
        density_norm = np.array([])
    else:
        density_range = np.max(densities) - np.min(densities)
        if density_range == 0:
            density_norm = np.zeros_like(densities)
        else:
            density_norm = (densities - np.min(densities)) / density_range

    print(f"mask_num for trace {ep_idx}:", mask_num)
    print(f"Trace end for trace {ep_idx}:", trace_end)

    return {
        "ep_idx": ep_idx,
        "trace_path": trace_path,
        "endpt_entry": endpt_entry,
        "density_norm": density_norm,
        "mask_num": mask_num,
    }


def get_trace_list_colored_parallel(img, endpts, tracer, viz=False, bimanual_declutter=False):
    """
    Parallel version of get_trace_list_colored.

    Traces each endpoint in a separate thread. The tracer's bool_detic_mask and
    detic_mask are read-only during tracing, so threading is safe.
    """
    img_up = img.copy()
    img_half = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    learned_tracer = tracer.tracer
    analytic_tracer = tracer.analytic_tracer
    L = len(endpts)
    min_condition_pts = getattr(getattr(learned_tracer, "trace_config", None), "condition_len", 3)

    # Launch all endpoint traces in parallel
    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=L) as executor:
        futures = {
            executor.submit(
                _trace_single_endpoint,
                ep_idx, img_up, endpts, learned_tracer, analytic_tracer, min_condition_pts
            ): ep_idx
            for ep_idx in range(L)
        }
        for future in as_completed(futures):
            res = future.result()
            results_by_idx[res["ep_idx"]] = res

    # Assemble results in endpoint order
    trace_list = []
    endpt_table = []
    densities_nrm = []
    mask_hit_list = []

    for ep_idx in range(L):
        res = results_by_idx[ep_idx]
        trace_list.append(res["trace_path"])
        endpt_table.append(res["endpt_entry"])
        densities_nrm.append(res["density_norm"])
        mask_hit_list.append(res["mask_num"])

    # Mask hit post-processing (same logic as original)
    print(f"Mask hit list: {mask_hit_list}")
    mask_hit_list = np.array([int(mask) - 1 for mask in mask_hit_list if mask != -1])

    if not bimanual_declutter:
        mask_hit = np.argwhere(mask_hit_list > -1)
        if len(mask_hit) > 0:
            mask_hit_list = mask_hit_list[mask_hit[0]]
        else:
            mask_hit_list = -1

    return trace_list, endpt_table, densities_nrm, mask_hit_list


def run_vision_parallel(img_rgb, img_down, detic_loader, endpt_model, tracer, viz=False):
    """
    Run Detic + endpoint detection in parallel, then parallel tracing.

    Returns:
        (bool_detic_mask, detic_mask, detic_out, endpoints,
         trace_list, endpt_table, densities, mask_num)
    """
    t0 = time.time()

    # Step 1: Detic and endpoint detection sequentially (CUDA models are not thread-safe)
    bool_detic_mask, detic_mask, detic_out = get_detic_masks(img_rgb, detic_loader)

    # Endpoint detector expects grayscale (3-channel) with contrast enhancement
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    alpha = 1.5
    beta = -20
    img_gray = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    endpoints = predict_pts(img_gray, endpt_model, bool_detic_mask, detic_mask)

    t1 = time.time()
    print(f"[parallel] Detic + endpoint detection: {t1 - t0:.2f}s")
    print(f"[parallel] Detected {len(endpoints)} endpoints")

    # Step 2: Set up tracer masks (must happen before tracing)
    tracer.tracer.bool_detic_mask = dilate_masks(bool_detic_mask, 15)
    tracer.tracer.detic_mask = dilate_masks(detic_mask, 15).clone()

    # Step 3: Parallel per-endpoint tracing (needs full-res image; get_mask halves internally)
    t2 = time.time()
    trace_list, endpt_table, densities, mask_num = get_trace_list_colored_parallel(
        img_rgb, endpoints, tracer, viz=viz,
        bimanual_declutter=BIMANUAL_DECLUTTER
    )
    t3 = time.time()
    print(f"[parallel] Tracing ({len(endpoints)} endpoints): {t3 - t2:.2f}s")
    print(f"[parallel] Total vision pipeline: {t3 - t0:.2f}s")

    return bool_detic_mask, detic_mask, detic_out, endpoints, trace_list, endpt_table, densities, mask_num


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------

def run_robot_pipeline(num_endpoints, output_dir=None, viz=False, dashboard=False):
    """Main robot pipeline loop with parallelized vision."""

    # Create output directory
    timestamp = datetime.now()
    output_dir = (output_dir and f"{output_dir}/{timestamp}") or f"{SAVE_DIR}/DATA_IROS26/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Launch dashboard as a background subprocess (avoids matplotlib backend conflict)
    if dashboard:
        dashboard_script = os.path.join(os.path.dirname(__file__), 'analysis_dashboard.py')
        subprocess.Popen(
            [sys.executable, dashboard_script, '--output_dir', output_dir, '--refresh', '2.0'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Dashboard launched, monitoring: {output_dir}")

    # Initialize logging
    log_path = os.path.join(output_dir, "metrics.txt")
    with open(log_path, "w") as f:
        f.write(f"Robot Pipeline Run (parallel): {timestamp}\n")
        f.write(f"Endpoints: {num_endpoints}\n\n")

    # Initialize camera first so we can show a live image immediately
    print("Initializing camera...")
    cam = BRIOSensor(0)
    print("Camera initialized.")

    # Capture and display a startup image before models load
    iter_0_path = os.path.join(output_dir, "iter_0")
    os.makedirs(iter_0_path, exist_ok=True)
    startup_img = capture_image(cam)
    cv2.imwrite(os.path.join(iter_0_path, "raw_image.png"), startup_img)
    write_status(output_dir, "Iter 0: Image captured — initializing models...")

    # Initialize robot
    print("Initializing robot...")
    interface = initialize_robot()
    print("Robot initialized.")

    # Initialize vision models
    print("Initializing vision models...")
    endpt_model = EndpointDataloader(use_hub_detect=True)
    detic_loader = DeticDataloader()
    detic_loader.create()
    detic_loader.default_vocab()
    tracer = TracerDataloader()
    print("Vision models initialized.")

    # Main loop
    for i in range(NUM_MOVES):
        print(f"\n=== Iteration {i} ===")
        iter_path = os.path.join(output_dir, f"iter_{i}")
        os.makedirs(iter_path, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(f"Iter {i}:\n")

        # Capture image
        write_status(output_dir, f"Iter {i}: Capturing image...")
        img = capture_image(cam)
        cv2.imwrite(os.path.join(iter_path, "raw_image.png"), img)
        write_status(output_dir, f"Iter {i}: Image captured — running vision pipeline...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_down = cv2.resize(img_rgb, (img_rgb.shape[1]//2, img_rgb.shape[0]//2))

        # Run parallelized vision pipeline
        write_status(output_dir, f"Iter {i}: Running Detic + endpoint detection in parallel...")
        (bool_detic_mask, detic_mask, detic_out, endpoints,
         trace_list, endpt_table, densities, mask_num) = run_vision_parallel(
            img_rgb, img_down, detic_loader, endpt_model, tracer, viz=viz
        )
        write_status(output_dir, f"Iter {i}: Tracing complete — {len(endpoints)} endpoints detected.")

        # Save visualizations
        save_object_masks_image(img_down, detic_mask, iter_path, viz)
        save_trace_images(img_down, trace_list, endpoints, {}, iter_path, viz)

        # Save endpoint detection visualization
        img_endpts = img_down.copy()
        for ep in endpoints:
            cv2.circle(img_endpts, (ep[1], ep[0]), 8, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(iter_path, 'endpoints.png'), cv2.cvtColor(img_endpts, cv2.COLOR_RGB2BGR))

        if len(endpoints) != num_endpoints:
            print(f"Warning: Expected {num_endpoints} endpoints, got {len(endpoints)}")

        # Filter for bimanual grasping if enabled
        if BIMANUAL_DECLUTTER:
            mask_num = bimanual_filter_grasps(mask_num, detic_out)

        # Skip masks that are near cable endpoints (hub attachments)
        if mask_num != -1:
            mask_num = filter_hub_masks(mask_num, detic_out, endpoints)

        # Decision: Declutter or IP maneuver
        if mask_num != -1:
            print("Executing declutter...")
            write_status(output_dir, f"Iter {i}: Executing declutter (removing objects)...")
            with open(log_path, "a") as f:
                f.write("Decluttering Objects\n")
            execute_declutter(interface, detic_out, mask_num)
            continue

        # Find divergence points
        write_status(output_dir, f"Iter {i}: Finding divergence points...")
        output = get_all_div_points(img_down, endpoints, trace_list, endpt_table, densities)
        asc_endpts = output.get('asc_endpts', [])

        # Check if all endpoints matched
        if len(asc_endpts) == num_endpoints // 2:
            write_status(output_dir, "All endpoints matched! Pipeline complete.")
            print("All endpoints matched! Pipeline complete.")
            break

        # No divergence points found
        if "div_points" not in output:
            write_status(output_dir, f"Iter {i}: No divergence points found — stopping.")
            print("No divergence points found.")
            break

        # Execute IP maneuver
        div_points = output["div_points"]
        trace_list = output["trace_list"]
        densities_pad = [np.concatenate((np.zeros(4), d)) for d in output["densities"]]
        endpt_1 = output["start_idx"]
        endpt_2 = output["end_idx"]

        do_push = True
        for div_pt in div_points:
            print("Classifying divergence point...")
            dense_count = classify_point_in_knot(img_down, div_pt, trace_list, densities_pad, viz=viz)

            if dense_count > 0:
                centroid, cent_area = find_open_knot_regions(img_down, div_pt, viz=viz)
                if centroid is not None:
                    print("Executing knot dilation...")
                    write_status(output_dir, f"Iter {i}: Executing Knot Dilation ({dense_count} dense px, area={cent_area:.1f})...")
                    execute_knot_dilation(interface, centroid)
                    with open(log_path, "a") as f:
                        f.write(f"Knot Dilation: {dense_count} dense px, area={cent_area:.2f}\n")
                    do_push = False
                    break

        if do_push:
            print("Executing push-through...")
            write_status(output_dir, f"Iter {i}: Executing Push Through...")
            try:
                push_target, start_coord, vec_angle, vec_dist, selected_div_point = plan_push_through(
                    div_points, img_down, trace_list, endpt_1, endpt_2, img_rgb, iter_path
                )
                save_divergence_image(img_down, [selected_div_point], trace_list, iter_path)
                perform_push_through(push_target, start_coord, interface)
                write_status(output_dir, f"Iter {i}: Push Through complete (angle={vec_angle:.1f}°, dist={vec_dist:.1f}px).")
                with open(log_path, "a") as f:
                    f.write(f"Push Through: angle={vec_angle:.2f}, dist={vec_dist:.2f}\n")
            except Exception as e:
                print(f"Push-through failed: {e}")
                write_status(output_dir, f"Iter {i}: Push Through FAILED — {e}")
                with open(log_path, "a") as f:
                    f.write(f"Push Through FAILED: {e}\n")
                break

    write_status(output_dir, f"Pipeline complete. Output saved to: {output_dir}")
    print(f"\nPipeline complete. Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Robot decluttering pipeline (parallelized vision)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decluttering_pipeline_robot_parallel.py --tier 2
  python decluttering_pipeline_robot_parallel.py --tier 4 --output_dir ./output --viz
        """
    )
    parser.add_argument('--tier', type=int, required=True,
                        help='Tier level (1-4): tier 1/2 = 4 endpoints, tier 3 = 6 endpoints, tier 4 = 8 endpoints')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for images/logs')
    parser.add_argument('--viz', action='store_true',
                        help='Show interactive matplotlib plots')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch live analysis dashboard (auto-refreshes every 2s)')
    args = parser.parse_args()

    # Validate tier
    if args.tier not in TIER_TO_ENDPOINTS:
        raise ValueError(f"tier must be 1, 2, 3, or 4, got {args.tier}")

    num_endpoints = TIER_TO_ENDPOINTS[args.tier]

    run_robot_pipeline(num_endpoints, args.output_dir, args.viz, args.dashboard)


if __name__ == '__main__':
    main()
