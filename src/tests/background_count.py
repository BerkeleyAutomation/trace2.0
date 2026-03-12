import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import cv2
import numpy as np
import torch
from utils.endpoint_detector_dataloader import EndpointDataloader
from utils.detic_dataloader import DeticDataloader
from masker import get_mask

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate what fraction of the background is filled with dark/black objects."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save visualization (default: current dir)")
    args = parser.parse_args()
    
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not load image: {args.image}")
    # Convert to 1-channel grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back to 3-channel (Gray, Gray, Gray)


    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    alpha = 1.5 
    beta = -20  # Pushes the darks lower

    # cv2.convertScaleAbs handles the rounding and clipping to [0, 255]
    img_gray = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    endpt_model = EndpointDataloader(use_hub_detect=True)

    detic_loader = DeticDataloader()
    detic_loader.create()
    detic_loader.default_vocab()

    bool_detic_mask, detic_mask, detic_out = get_detic_masks(img_gray, detic_loader)

    endpoints = predict_pts(img_gray, endpt_model, bool_detic_mask, detic_mask)

    # Union cable masks from all detected endpoints
    combined_cable_mask = np.zeros((img.shape[0] // 2, img.shape[1] // 2), dtype=np.uint8)
    for endpt in endpoints:
        combined_cable_mask = cv2.bitwise_or(combined_cable_mask, get_mask(img, endpt))

    # Background = pixels not covered by any cable mask
    background_mask = combined_cable_mask == 0

    # Estimate background cloth color from image corners in LAB space.
    # Corners are most likely to be pure background (no cables or objects).
    lab_down = cv2.cvtColor(img_down, cv2.COLOR_BGR2LAB)
    border = max(20, min(img_down.shape[0], img_down.shape[1]) // 20)
    corner_pixels = np.vstack([
        lab_down[:border, :border].reshape(-1, 3),
        lab_down[:border, -border:].reshape(-1, 3),
        lab_down[-border:, :border].reshape(-1, 3),
        lab_down[-border:, -border:].reshape(-1, 3),
    ])
    bg_color_lab = np.median(corner_pixels, axis=0)

    # Per-pixel Euclidean distance from background color in LAB space.
    diff = lab_down.astype(np.float32) - bg_color_lab.astype(np.float32)
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Smooth to reduce noise, then use Otsu's method — but only on pixels that
    # are already in the background region (non-cable areas).  Running Otsu on
    # the full image would just find the cable/background split; restricting to
    # the background region finds the split between cloth and true anomalies.
    color_dist_blurred = cv2.GaussianBlur(color_dist, (5, 5), 0)
    bg_dist_values = color_dist_blurred[background_mask]
    if len(bg_dist_values) > 0 and bg_dist_values.std() > 1:
        bg_dist_u8 = np.clip(bg_dist_values, 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu_thresh, _ = cv2.threshold(bg_dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        otsu_thresh = 255  # nothing looks like clutter
    not_background = color_dist_blurred > otsu_thresh

    # Morphological opening removes small speckles; closing fills small holes in objects.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    not_bg_clean = cv2.morphologyEx(not_background.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    not_bg_clean = cv2.morphologyEx(not_bg_clean, cv2.MORPH_CLOSE, kernel)

    # Clutter = pixels that (a) are not covered by a detected cable and (b) don't
    # look like the background cloth colour.
    near_black = not_bg_clean.astype(bool) == 0

    background_pixel_count = np.sum(background_mask)
    black_in_background = np.sum(background_mask & near_black)

    pct = (black_in_background / background_pixel_count * 100) if background_pixel_count > 0 else 0.0
    print(f"Background clutter pixel percentage: {pct:.2f}%")
    print(f"  (Otsu LAB-distance threshold: {otsu_thresh:.1f}, background LAB: {bg_color_lab})")

    # Build clutter mask image (white where clutter detected, black elsewhere)
    gray_down = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)
    black_bg_mask_img = np.zeros_like(gray_down)
    black_bg_mask_img[background_mask & near_black] = 255

    # Side-by-side visualization: raw image | black background mask
    mask_rgb = cv2.cvtColor(black_bg_mask_img, cv2.COLOR_GRAY2BGR)
    vis = np.concatenate([img_down, mask_rgb], axis=1)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.output_dir, f"{stem}_background_count.png")
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization to {out_path}")
