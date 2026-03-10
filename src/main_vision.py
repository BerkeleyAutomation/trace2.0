"""
Vision-only cable tracing pipeline with parallelized vision steps.

Same as decluttering_pipeline_vision_colored.py but runs:
  1. Detic object detection + endpoint detection sequentially (CUDA models not thread-safe)
  2. Per-endpoint cable tracing in parallel

Takes a top-down image as input and outputs cable trace visualizations.
No robot code - purely vision processing.

Usage:
    python decluttering_pipeline_vision_parallel.py --image <path> --tier <1-4> [--output_dir <dir>] [--viz]
"""

# Tier to endpoints mapping
TIER_TO_ENDPOINTS = {1: 4, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16}

import argparse
import copy
import cv2
import numpy as np
import math
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.detic_dataloader import DeticDataloader
from utils.tracer_dataloader import TracerDataloader
from utils.endpoint_detector_dataloader import EndpointDataloader
from divergencecopy import get_all_div_points, visualize_multiple_paths
from run_constants import COLORS, SAVE_DIR

# Per-endpoint trace helpers
from utils.tracer.tusk_pipeline.tracer import TraceEnd
from masker import get_mask


# ---------------------------------------------------------------------------
# Vision helper functions (from decluttering_pipeline_vision_colored.py)
# ---------------------------------------------------------------------------

def dilate_masks(mask_tensor, kernel_size=3):
    """Dilate masks using max pooling."""
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(0)

    mask_tensor = mask_tensor.float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask_tensor.device)

    dilated = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)

    if mask_tensor.dtype == torch.bool:
        dilated = dilated > 0

    dilated = dilated.squeeze()
    return dilated


def get_detic_masks(img, detic):
    """Run Detic object detection and return masks."""
    out = detic.predict(img)

    bool_arr = []
    for i in range(out['boxes'].shape[0]):
        if out['boxes'][i][0] > 100 and out['boxes'][i][2] < 3500:
            bool_arr.append(True)
        else:
            bool_arr.append(False)
    out['boxes'] = out['boxes'][bool_arr]
    out['masks'] = out['masks'][bool_arr]
    out['masks'] = detic.filter_mask_by_area(out['masks'], max_area=250000)

    img_height, img_width = out['masks'].shape[2], out['masks'].shape[3]
    detic_masked_arr = torch.zeros((img_height, img_width), dtype=torch.uint8).to('cuda')
    bool_detic_masked_arr = torch.zeros((img_height, img_width), dtype=bool).to('cuda')
    masks = out['masks']

    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            detic_masked_arr = torch.maximum(detic_masked_arr, masks[i][j] * (i + 1))
            bool_detic_masked_arr = torch.logical_or(bool_detic_masked_arr, masks[i][j])

    return bool_detic_masked_arr, detic_masked_arr, out


def predict_pts(img, endpt_model, bool_detic_mask=None, detic_mask=None):
    """Detect cable endpoints in the image."""
    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    endpoints = endpt_model.predict(img_down)['endpoints']

    if bool_detic_mask is not None:
        filtered_endpoints = []
        for endpt in endpoints:
            if bool_detic_mask[endpt[0]*2][endpt[1]*2] == 0:
                filtered_endpoints.append(endpt)
        return filtered_endpoints

    return endpoints


def save_object_masks_image(img, detic_mask, output_dir, viz):
    """Save original image with object detection masks overlaid."""
    img_with_masks = img.copy()

    # Resize mask to match downsampled image
    mask_np = detic_mask.cpu().numpy().astype('uint8')
    mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create colored overlay for each unique mask value
    mask_overlay = np.zeros_like(img)
    unique_vals = np.unique(mask_resized)
    for val in unique_vals:
        if val == 0:
            continue
        color_idx = (val - 1) % len(COLORS)
        mask_overlay[mask_resized == val] = COLORS[color_idx]

    img_with_masks = cv2.addWeighted(img_with_masks, 0.6, mask_overlay, 0.4, 0)

    output_path = os.path.join(output_dir, 'object_masks.png')
    cv2.imwrite(output_path, cv2.cvtColor(img_with_masks, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")

    if viz:
        plt.figure(figsize=(16, 12))
        plt.imshow(img_with_masks)
        plt.title('Object Detection Masks')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def save_trace_images(img, trace_list, endpoints, output, output_dir, viz):
    """Save individual and combined trace visualizations."""

    # Combined trace image (all cables, different colors)
    combined = visualize_multiple_paths(img.copy(), trace_list, [COLORS[i % len(COLORS)] for i in range(len(trace_list))])

    output_path = os.path.join(output_dir, 'combined_traces.png')
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")

    # Individual trace images (one per cable)
    for i, trace in enumerate(trace_list):
        single = visualize_multiple_paths(img.copy(), [trace], [COLORS[i % len(COLORS)]])
        output_path = os.path.join(output_dir, f'trace_{i}.png')
        cv2.imwrite(output_path, cv2.cvtColor(single, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

    if viz:
        plt.figure(figsize=(16, 12))
        plt.imshow(combined)
        plt.title(f'{len(trace_list)} Cable Traces')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Parallel vision helpers (from decluttering_pipeline_robot_parallel.py)
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
    Run Detic + endpoint detection sequentially, then parallel tracing.

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
        bimanual_declutter=False
    )
    t3 = time.time()
    print(f"[parallel] Tracing ({len(endpoints)} endpoints): {t3 - t2:.2f}s")
    print(f"[parallel] Total vision pipeline: {t3 - t0:.2f}s")

    return bool_detic_mask, detic_mask, detic_out, endpoints, trace_list, endpt_table, densities, mask_num


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_vision_pipeline(image_path, output_dir, num_endpoints, viz=False):
    """
    Run the full parallelized vision pipeline on an input image.

    Args:
        image_path: Path to input image
        output_dir: Directory to save output images
        num_endpoints: Expected number of endpoints
        viz: Whether to show interactive plots

    Returns:
        dict with trace_list, endpoints, divergence_points, etc.
    """
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_down = cv2.resize(img_rgb, (img_rgb.shape[1]//2, img_rgb.shape[0]//2))

    print("Initializing models...")
    endpt_model = EndpointDataloader(use_hub_detect=True)
    detic_loader = DeticDataloader()
    detic_loader.create()
    detic_loader.default_vocab()
    tracer = TracerDataloader()
    print("Models initialized.")

    # Run parallelized vision pipeline
    (bool_detic_mask, detic_mask, detic_out, endpoints,
     trace_list, endpt_table, densities, mask_num) = run_vision_parallel(
        img_rgb, img_down, detic_loader, endpt_model, tracer, viz=viz
    )
    print(f"Detected {detic_out['masks'].shape[0]} objects")
    print(f"Detected {len(endpoints)} endpoints (expected {num_endpoints})")

    if len(endpoints) != num_endpoints:
        print(f"WARNING: Detected {len(endpoints)} endpoints, expected {num_endpoints}")

    # Save visualizations
    save_object_masks_image(img_down, detic_mask, output_dir, viz)

    print("Finding divergence points...")
    output = get_all_div_points(img_down, endpoints, trace_list, endpt_table, densities)

    if "div_points" in output:
        print(f"Found {len(output['div_points'])} divergence points")
    else:
        print("No divergence points found - traces complete")

    save_trace_images(img_down, trace_list, endpoints, output, output_dir, viz)

    print(f"\nOutput saved to: {output_dir}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Vision-only cable tracing pipeline (parallelized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decluttering_pipeline_vision_parallel.py --image input.png --tier 2
  python decluttering_pipeline_vision_parallel.py --image input.png --tier 4 --viz
        """
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--tier', type=int, required=True,
                        help='Tier level (1-4): tier 1/2 = 4 endpoints, tier 3 = 6 endpoints, tier 4 = 8 endpoints')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: DATA_IROS26/<timestamp>)')
    parser.add_argument('--viz', action='store_true',
                        help='Show interactive matplotlib plots')

    args = parser.parse_args()

    # Validate tier
    if args.tier not in TIER_TO_ENDPOINTS:
        print(f"Error: tier must be 1-8, got {args.tier}")
        sys.exit(1)

    num_endpoints = TIER_TO_ENDPOINTS[args.tier]

    # Validate input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Set output directory (default: DATA_IROS26/<timestamp>)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(SAVE_DIR, "NEW_DATA", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Run pipeline
    run_vision_pipeline(args.image, output_dir, num_endpoints, args.viz)


if __name__ == '__main__':
    main()
