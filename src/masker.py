# import cv2
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# sys.path.append('/home/justinyu/multicable-decluttering/')


# def extract_cable_pixels(img, brightness_thresh=120):
#     """
#     Extract pixels that are likely cables (bright) against a dark background.
#     Uses a higher threshold to exclude background noise, then morphological
#     operations to focus on elongated cable-like structures.

#     Args:
#         img: RGB image (H, W, 3), uint8
#         brightness_thresh: minimum brightness (max of RGB channels) to be a cable pixel

#     Returns:
#         cable_mask: boolean (H, W), True where pixels are likely cables
#         cable_coords: (N, 2) array of (row, col) coordinates
#         cable_colors: (N, 3) array of RGB values
#     """
#     # Use max of RGB channels as brightness
#     brightness = np.max(img, axis=2)

#     # Initial threshold: only clearly bright pixels
#     bright_mask = brightness > brightness_thresh

#     # Convert to uint8 for morphological ops
#     mask_u8 = bright_mask.astype(np.uint8) * 255

#     # Morphological open to remove small noise specks
#     open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

#     # Remove large blobs (connectors, background patches) — keep only thin cable-like structures
#     # We use the ratio of perimeter^2 / area to identify elongated shapes
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
#     cable_only_mask = np.zeros_like(mask_u8)
#     for label_id in range(1, num_labels):
#         area = stats[label_id, cv2.CC_STAT_AREA]
#         # Skip very small noise
#         if area < 100:
#             continue
#         # Keep components regardless of shape — the KMeans should handle color separation
#         # But skip extremely large blobs (likely background patches or connectors)
#         if area > img.shape[0] * img.shape[1] * 0.1:
#             continue
#         cable_only_mask[labels == label_id] = 255

#     cable_mask = cable_only_mask > 0
#     cable_coords = np.argwhere(cable_mask)
#     cable_colors = img[cable_mask]

#     return cable_mask, cable_coords, cable_colors


# def cluster_cable_colors(cable_colors, num_cables):
#     """
#     Cluster cable pixels by color using KMeans in RGB space.
#     """
#     # Subsample for speed if too many pixels
#     max_samples = 50000
#     if len(cable_colors) > max_samples:
#         indices = np.random.choice(len(cable_colors), max_samples, replace=False)
#         sample_colors = cable_colors[indices]
#     else:
#         sample_colors = cable_colors

#     kmeans = KMeans(n_clusters=num_cables, random_state=42, n_init=10)
#     kmeans.fit(sample_colors.astype(np.float32))

#     # Predict labels for all cable pixels
#     labels = kmeans.predict(cable_colors.astype(np.float32))
#     centers = kmeans.cluster_centers_.astype(np.uint8)

#     return labels, centers


# def build_per_cable_masks(img, cable_coords, labels, centers, num_cables, rgb_margin=25):
#     """
#     Build per-cable masks. For each cluster, create a mask of all image pixels
#     that are closest (in RGB distance) to this cluster's center, among
#     pixels that are bright enough.
#     """
#     h, w = img.shape[:2]
#     masks = []

#     for k in range(num_cables):
#         cluster_pixels = cable_coords[labels == k]
#         if len(cluster_pixels) == 0:
#             masks.append(np.zeros((h, w), dtype=np.uint8))
#             continue

#         # Create mask from the clustered pixels
#         mask = np.zeros((h, w), dtype=np.uint8)
#         mask[cluster_pixels[:, 0], cluster_pixels[:, 1]] = 255
#         masks.append(mask)

#     return masks


# def clean_cable_mask(mask, min_component_area=200, close_kernel_size=9, dilate_iterations=3):
#     """
#     Clean up a raw cable mask:
#       1. Morphological close to bridge small gaps
#       2. Remove small isolated components
#       3. Dilation to thicken thin cable traces
#     """
#     if mask.max() == 0:
#         return mask

#     # Step 1: Morphological close (bridge small gaps)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
#     closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Step 2: Remove small connected components
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
#     cleaned = np.zeros_like(closed)
#     for label_id in range(1, num_labels):
#         if stats[label_id, cv2.CC_STAT_AREA] >= min_component_area:
#             cleaned[labels == label_id] = 255

#     # Step 3: Dilate to thicken cable traces
#     dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=dilate_iterations)

#     return cleaned


# def segment_cables_by_color(img, num_cables,
#                              brightness_thresh=120,
#                              rgb_margin=25,
#                              min_component_area=300,
#                              viz=False):
#     """
#     Full pipeline: extract bright cable pixels, cluster by color, build per-cable masks.
#     """
#     print("Extracting cable pixels from background...")
#     cable_mask_all, cable_coords, cable_colors = extract_cable_pixels(
#         img, brightness_thresh=brightness_thresh
#     )
#     print(f"  Found {len(cable_coords)} candidate cable pixels")

#     if len(cable_colors) == 0:
#         print("ERROR: No cable pixels found. Try lowering --brightness threshold.")
#         return [], np.array([])

#     print(f"Clustering into {num_cables} color groups...")
#     labels, centers = cluster_cable_colors(cable_colors, num_cables)

#     for i, center in enumerate(centers):
#         count = np.sum(labels == i)
#         print(f"  Cluster {i}: RGB=({center[0]}, {center[1]}, {center[2]}), {count} pixels")

#     print("Building per-cable masks...")
#     raw_masks = build_per_cable_masks(
#         img, cable_coords, labels, centers, num_cables, rgb_margin
#     )

#     print("Cleaning masks...")
#     cable_masks = []
#     for i, raw_mask in enumerate(raw_masks):
#         cleaned = clean_cable_mask(raw_mask, min_component_area=min_component_area)
#         cable_masks.append(cleaned)
#         print(f"  Cable {i}: RGB=({centers[i][0]}, {centers[i][1]}, {centers[i][2]}), "
#               f"mask pixels={np.sum(cleaned > 0)}")

#     if viz:
#         _visualize_segmentation(img, cable_masks, centers, cable_mask_all)

#     return cable_masks, centers


# def _visualize_segmentation(img, cable_masks, centers, all_cable_mask):
#     """Show diagnostic plots for the segmentation."""
#     n = len(cable_masks)
#     fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 10))

#     if n + 1 == 1:
#         axes = axes.reshape(2, 1)

#     # Top-left: original image
#     axes[0, 0].imshow(img)
#     axes[0, 0].set_title('Original Image')
#     axes[0, 0].axis('off')

#     # Bottom-left: all cable pixels (pre-clustering)
#     axes[1, 0].imshow(all_cable_mask, cmap='gray')
#     axes[1, 0].set_title(f'All Cable Pixels\n({np.sum(all_cable_mask)} px)')
#     axes[1, 0].axis('off')

#     # Per-cable views
#     VIZ_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
#                   (255,0,255), (0,255,255), (255,128,0), (128,0,255)]
#     for i in range(n):
#         # Top row: overlay on original
#         overlay = img.copy()
#         color = VIZ_COLORS[i % len(VIZ_COLORS)]
#         colored = np.zeros_like(img)
#         colored[cable_masks[i] > 0] = color
#         overlay = cv2.addWeighted(overlay, 0.7, colored, 0.5, 0)
#         axes[0, i + 1].imshow(overlay)
#         axes[0, i + 1].set_title(f'Cable {i}\nRGB=({centers[i][0]},{centers[i][1]},{centers[i][2]})')
#         axes[0, i + 1].axis('off')

#         # Bottom row: binary mask
#         axes[1, i + 1].imshow(cable_masks[i], cmap='gray')
#         axes[1, i + 1].set_title(f'Cable {i} Mask\n({np.sum(cable_masks[i] > 0)} px)')
#         axes[1, i + 1].axis('off')

#     plt.tight_layout()
#     plt.show()


# def make_white_cable_image(img, cable_mask):
#     """
#     Convert a single cable's mask into a white-on-black image
#     that the existing tracer can consume directly.
#     """
#     white_cable_img = np.zeros_like(img)
#     white_cable_img[cable_mask > 0] = 255
#     return white_cable_img


# if __name__ == "__main__":
#     import argparse
#     import os
#     from datetime import datetime

#     TIER_TO_ENDPOINTS = {1: 4, 2: 4, 3: 6, 4: 8}

#     parser = argparse.ArgumentParser(
#         description='Color-based cable segmentation test',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python colored_decluttering_test.py --image input.png --tier 3
#   python colored_decluttering_test.py --image input.png --tier 4 --viz
#   python colored_decluttering_test.py --image input.png --tier 2 --brightness 100 --rgb_margin 20
#         """
#     )
#     parser.add_argument('--image', type=str, required=True, help='Path to input image')
#     parser.add_argument('--tier', type=int, required=True,
#                         help='Tier level (1-4): determines expected number of cables')
#     parser.add_argument('--brightness', type=int, default=120,
#                         help='Min brightness (max of RGB) to consider a pixel as cable (default: 120)')
#     parser.add_argument('--rgb_margin', type=int, default=25,
#                         help='Extra margin on each side of the RGB cube (default: 25)')
#     parser.add_argument('--min_area', type=int, default=300,
#                         help='Min connected component area to keep (default: 300)')
#     parser.add_argument('--output_dir', type=str, default=None,
#                         help='Output directory (default: auto-generated timestamp dir)')
#     parser.add_argument('--viz', action='store_true', help='Show interactive matplotlib plots')
#     parser.add_argument('--downsample', action='store_true',
#                         help='Downsample image by 2x before processing')

#     args = parser.parse_args()

#     if args.tier not in TIER_TO_ENDPOINTS:
#         print(f"Error: tier must be 1, 2, 3, or 4, got {args.tier}")
#         exit(1)

#     num_endpoints = TIER_TO_ENDPOINTS[args.tier]
#     num_cables = num_endpoints // 2

#     if not os.path.exists(args.image):
#         print(f"Error: Image not found: {args.image}")
#         exit(1)

#     # Set up output directory
#     if args.output_dir:
#         output_dir = args.output_dir
#     else:
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         output_dir = os.path.join(os.path.dirname(os.path.abspath(args.image)), f"color_seg_{timestamp}")
#     os.makedirs(output_dir, exist_ok=True)

#     # Load image
#     print(f"Loading image: {args.image}")
#     img = cv2.imread(args.image)
#     if img is None:
#         print(f"Error: Could not load image: {args.image}")
#         exit(1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     if args.downsample:
#         img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
#         print(f"Downsampled to {img.shape[1]}x{img.shape[0]}")

#     print(f"Image size: {img.shape[1]}x{img.shape[0]}")
#     print(f"Num cables: {num_cables}")
#     print(f"Brightness thresh: {args.brightness}, RGB margin: {args.rgb_margin}, Min area: {args.min_area}")

#     # Run color segmentation
#     cable_masks, centers = segment_cables_by_color(
#         img, num_cables,
#         brightness_thresh=args.brightness,
#         rgb_margin=args.rgb_margin,
#         min_component_area=args.min_area,
#         viz=args.viz
#     )

#     print(f"\nSegmented {len(cable_masks)} cables. Saving to {output_dir}...")

#     # Save per-cable masks and white-on-black images
#     for i, mask in enumerate(cable_masks):
#         mask_path = os.path.join(output_dir, f"cable_{i}_mask.png")
#         cv2.imwrite(mask_path, mask)
#         print(f"  Saved mask: {mask_path}")

#         white_img = make_white_cable_image(img, mask)
#         white_path = os.path.join(output_dir, f"cable_{i}_white.png")
#         cv2.imwrite(white_path, cv2.cvtColor(white_img, cv2.COLOR_RGB2BGR))
#         print(f"  Saved white-on-black: {white_path}")

#     # Save combined overlay
#     VIZ_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
#                   (255,0,255), (0,255,255), (255,128,0), (128,0,255)]
#     overlay = img.copy()
#     for i, mask in enumerate(cable_masks):
#         color = VIZ_COLORS[i % len(VIZ_COLORS)]
#         colored = np.zeros_like(img)
#         colored[mask > 0] = color
#         overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
#     overlay_path = os.path.join(output_dir, "combined_overlay.png")
#     cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
#     print(f"  Saved overlay: {overlay_path}")

#     print(f"\nDone! All outputs in: {output_dir}")

import sys
import cv2
import torch
import numpy as np
import argparse
from utils.endpoint_detector_dataloader import EndpointDataloader
from utils.detic_dataloader import DeticDataloader
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


def sample_color(img, endpoint):

    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    # Offset from the center
    offset_w, offset_h = 5, 5
    x, y = int(endpoint[1]), int(endpoint[0]) # Ensure they are integers
    
    # Get image dimensions (H, W, C)
    img_h, img_w = img_down.shape[:2]

    # Calculate bounds and clip them to stay within image boundaries
    y_min = max(0, y - offset_h)
    y_max = min(img_h, y + offset_h + 1)
    x_min = max(0, x - offset_w)
    x_max = min(img_w, x + offset_w + 1)

    # Slice the image to get the region of interest (ROI)
    roi = img_down[y_min:y_max, x_min:x_max]

    # Flatten the ROI into a list of pixel values
    # If it's a 3-channel image, each element is [R, G, B]
    pixel_list = roi.reshape(-1, img_down.shape[2]).tolist()

    return pixel_list

def create_color_mask(img, color_list, buffer=10):

    """

    Creates a binary mask where pixels falling within the range 

    of the provided color_list (plus/minus a buffer) are white.

    """

    # Convert list to a numpy array for easy min/max calculation

    # img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

    colors = np.array(color_list)
    if colors.size == 0:
        # No sampled colors; return an empty mask at downsampled tracer resolution.
        return np.zeros((img.shape[0] // 2, img.shape[1] // 2), dtype=np.uint8)

    

    # Find the min and max for each channel (R, G, B)

    # We add a small buffer so the mask isn't "perfectly" strict

    min_color = np.min(colors, axis=0) - buffer

    max_color = np.max(colors, axis=0) + buffer

    

    # Clip values to stay within the valid [0, 255] range

    min_color = np.clip(min_color, 0, 255)

    max_color = np.clip(max_color, 0, 255)

    

    # Create the mask (255 where pixels are in range, 0 otherwise)

    mask = cv2.inRange(img, min_color, max_color)

    

    kernel_noise = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise) # Remove noise
    kernel_fill = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill) # Fill holes
    # kernel_fill_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill_ellipse) # Fill holes
    mask = cv2.resize(mask, (img.shape[1]//2, img.shape[0]//2))
    return mask

def sample_color_hsv(img, endpoint):
    """Converts the ROI to HSV and samples the pixels."""
    # 1. Convert the whole image or just the crop to HSV
    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    img_hsv = cv2.cvtColor(img_down, cv2.COLOR_RGB2HSV)
    
    offset = 5
    x, y = int(endpoint[1]), int(endpoint[0])
    h, w = img_hsv.shape[:2]

    y_min, y_max = max(0, y - offset), min(h, y + offset + 1)
    x_min, x_max = max(0, x - offset), min(w, x + offset + 1)

    roi_hsv = img_hsv[y_min:y_max, x_min:x_max]
    
    if roi_hsv.size == 0:
        return []
    
    return roi_hsv.reshape(-1, 3).tolist()

def create_hsv_mask2(img, color_list_hsv, h_buf=5, s_buf=5, v_buf=5):

    """
    Creates a mask using HSV ranges. 
    H_buf is tight (color), S and V buffers are loose (lighting/intensity).
    """
    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    img_hsv = cv2.cvtColor(img_down, cv2.COLOR_RGB2HSV)
    colors = np.array(color_list_hsv)
    median_hsv = np.min(colors, axis=0)
    lower_hsv = np.array([
        median_hsv[0] - h_buf, 
        median_hsv[1] - s_buf, 
        median_hsv[2] - v_buf
    ])
    median_hsv = np.max(colors, axis=0)
    upper_hsv = np.array([
        median_hsv[0] + h_buf, 
        median_hsv[1] + s_buf, 
        median_hsv[2] + v_buf
    ]) 
    print("AAAA", lower_hsv, upper_hsv)


    lower_h, lower_s, lower_v = lower_hsv
    upper_h, upper_s, upper_v = upper_hsv

    if lower_h >= 0 and upper_h <= 180:
        mask = cv2.inRange(img_hsv, np.array([lower_h, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))
    
    # Wrapping case: Hue < 0
    if lower_h < 0:
        mask1 = cv2.inRange(img_hsv, np.array([180 + lower_h, lower_s, lower_v]), np.array([180, upper_s, upper_v]))
        mask2 = cv2.inRange(img_hsv, np.array([0, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))
        mask = cv2.bitwise_or(mask1, mask2)
        
    # Wrapping case: Hue > 180
    if upper_h > 180:
        mask1 = cv2.inRange(img_hsv, np.array([lower_h, lower_s, lower_v]), np.array([180, upper_s, upper_v]))
        mask2 = cv2.inRange(img_hsv, np.array([0, lower_s, lower_v]), np.array([upper_h - 180, upper_s, upper_v]))
        mask = cv2.bitwise_or(mask1, mask2)

    
    kernel_noise = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise) # Remove noise
    kernel_fill = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill) # Fill holes
    # kernel_fill_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill_ellipse) # Fill holes
    
    return mask
def create_hsv_mask(img, color_list_hsv, h_buf=40, s_buf=80, v_buf=100):

    """
    Creates a mask using HSV ranges. 
    H_buf is tight (color), S and V buffers are loose (lighting/intensity).
    """
    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    img_hsv = cv2.cvtColor(img_down, cv2.COLOR_RGB2HSV)
    colors = np.array(color_list_hsv)
    if colors.size == 0:
        return np.zeros((img_down.shape[0], img_down.shape[1]), dtype=np.uint8)
    
    # Calculate medians for a more stable center point than min/max
    median_hsv = np.median(colors, axis=0)
    
    # Define bounds: [Hue, Saturation, Value]
    # Hue wraps at 180 in OpenCV, but for simple masking we clip
    lower_hsv = np.array([
        median_hsv[0] - h_buf, 
        median_hsv[1] - s_buf, 
        median_hsv[2] - v_buf
    ])
    upper_hsv = np.array([
        median_hsv[0] + h_buf, 
        median_hsv[1] + s_buf, 
        median_hsv[2] + v_buf
    ])

    lower_h, lower_s, lower_v = lower_hsv
    upper_h, upper_s, upper_v = upper_hsv

    if lower_h >= 0 and upper_h <= 180:
        mask = cv2.inRange(img_hsv, np.array([lower_h, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))
    
    # Wrapping case: Hue < 0
    if lower_h < 0:
        mask1 = cv2.inRange(img_hsv, np.array([180 + lower_h, lower_s, lower_v]), np.array([180, upper_s, upper_v]))
        mask2 = cv2.inRange(img_hsv, np.array([0, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))
        mask = cv2.bitwise_or(mask1, mask2)
        
    # Wrapping case: Hue > 180
    if upper_h > 180:
        mask1 = cv2.inRange(img_hsv, np.array([lower_h, lower_s, lower_v]), np.array([180, upper_s, upper_v]))
        mask2 = cv2.inRange(img_hsv, np.array([0, lower_s, lower_v]), np.array([upper_h - 180, upper_s, upper_v]))
        mask = cv2.bitwise_or(mask1, mask2)

    # Ensure bounds are within [0, 180] for H and [0, 255] for S, V
    # lower_hsv = np.clip(lower_hsv, [0, 5, 5], [180, 255, 255])
    # upper_hsv = np.clip(upper_hsv, [0, 5, 5], [180, 255, 255])
    
    # mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
    # Post-processing: remove small noise and fill holes
    kernel_noise = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise) # Remove noise
    kernel_fill = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill) # Fill holes
    # kernel_fill_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill_ellipse) # Fill holes
    
    return mask

def get_mask(img, end_pt):
    color_list_hsv = sample_color_hsv(img, end_pt)
    color_list_rgb = sample_color(img, end_pt)
    mask = cv2.bitwise_or(create_color_mask(img, color_list_rgb), create_hsv_mask(img, color_list_hsv))
    return mask


def get_intersection(box1, box2):
    # box = [lower_bound, upper_bound] where bound is [R, G, B]
    low = np.maximum(box1[0], box2[0])
    high = np.minimum(box1[1], box2[1])
    if np.all(low <= high):
        return low, high
    return None

def resolve_overlap(static_box, moving_box):
    intersection = get_intersection(static_box, moving_box)
    if not intersection:
        return moving_box
    
    # Calculate overlap depth on each axis
    low_int, high_int = intersection
    overlap_depths = high_int - low_int
    
    # Find the axis with the smallest overlap to minimize "shrinkage"
    axis = np.argmin(overlap_depths)
    
    new_moving = np.copy(moving_box)
    # If the moving box's max is inside the static box, push its max down
    if moving_box[1][axis] <= static_box[1][axis]:
        new_moving[1][axis] = static_box[0][axis] - 1
    # Otherwise, push its min up
    else:
        new_moving[0][axis] = static_box[1][axis] + 1
        
    return new_moving

def calculate_volume(range_pair):
    lower, upper = np.array(range_pair[0]), np.array(range_pair[1])
    diff = np.maximum(0, upper - lower)
    return np.prod(diff)
def resolve_all_overlaps(color_ranges):
    # Attach original index: (index, (low, high))
    indexed_ranges = list(enumerate(color_ranges))
    
    # Sort by volume but keep the index attached
    indexed_ranges.sort(key=lambda x: calculate_volume(x[1]), reverse=True)
    
    final_indexed_ranges = []
    for i, (original_idx, current_range) in enumerate(indexed_ranges):
        current_low = np.array(current_range[0], dtype=np.int16)
        current_high = np.array(current_range[1], dtype=np.int16)
        
        # Check against every range we've already "fixed"
        for fixed_idx, (fixed_low, fixed_high) in final_indexed_ranges:
            # Check for 3D intersection
            overlap_low = np.maximum(current_low, fixed_low)
            overlap_high = np.minimum(current_high, fixed_high)
            
            if np.all(overlap_low <= overlap_high):
                # We have an overlap. Shrink the current_range along 
                # the axis with the smallest overlap depth.
                depths = overlap_high - overlap_low + 1
                axis = np.argmin(depths)
                
                # If the current range is "mostly" higher than the fixed range on this axis
                if current_high[axis] > fixed_high[axis]:
                    current_low[axis] = fixed_high[axis] + 1
                else:
                    current_high[axis] = fixed_low[axis] - 1
                
                # Clamp values to valid RGB range [0, 255]
                current_low = np.clip(current_low, 0, 255)
                current_high = np.clip(current_high, 0, 255)
        
        # Only add if the range hasn't been shrunk into non-existence
        if np.all(current_low <= current_high):
            final_indexed_ranges.append((original_idx, (current_low, current_high)))
        else:
            final_indexed_ranges.append((original_idx, ([0, 0, 0], [0, 0, 0])))
            
    final_indexed_ranges.sort(key=lambda x: x[0])
    return [range_val for idx, range_val in final_indexed_ranges]

def get_all_masks(img, end_pts):
    hsv_colors, rgb_colors = [], []
    for pt in end_pts:
        hsv_colors.append(sample_color_hsv(img, pt))
        rgb_colors.append(sample_color(img, pt))

    rgb_buffer=20
    hsv_buffer=(20, 80, 100)
    rgb_ranges, hsv_ranges = [], []
    for i in range(len(end_pts)):
        colors = np.array(rgb_colors[i])
        min_color = np.min(colors, axis=0) - rgb_buffer
        max_color = np.max(colors, axis=0) + rgb_buffer
        min_color = np.clip(min_color, 0, 255)
        max_color = np.clip(max_color, 0, 255)
    
        rgb_ranges.append([min_color, max_color])
        median_hsv = np.median(hsv_colors[i], axis=0)
        
        # Define bounds: [Hue, Saturation, Value]
        # Hue wraps at 180 in OpenCV, but for simple masking we clip
        lower_hsv = np.array([
            median_hsv[0] - hsv_buffer[0], 
            median_hsv[1] - hsv_buffer[1], 
            median_hsv[2] - hsv_buffer[2]
        ])
        upper_hsv = np.array([
            median_hsv[0] + hsv_buffer[0], 
            median_hsv[1] + hsv_buffer[1], 
            median_hsv[2] + hsv_buffer[2]
        ])

        # Ensure bounds are within [0, 180] for H and [0, 255] for S, V
        lower_hsv = np.clip(lower_hsv, [0, 5, 5], [180, 255, 255])
        upper_hsv = np.clip(upper_hsv, [0, 5, 5], [180, 255, 255])
        hsv_ranges.append([lower_hsv, upper_hsv])

    print("pre_rgb_ranges", rgb_ranges)
    rgb_ranges = resolve_all_overlaps(rgb_ranges)
    print("postrgb_ranges", rgb_ranges)
    hsv_ranges = resolve_all_overlaps(hsv_ranges)

    masks = []
    for i in range(len(end_pts)):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        print(rgb_ranges[i])

        rgb_mask = cv2.inRange(img, np.array(rgb_ranges[i][0]), np.array(rgb_ranges[i][1]))
        hsv_mask = cv2.inRange(img_hsv, np.array(hsv_ranges[i][0]), np.array(hsv_ranges[i][1]))

        kernel_noise = np.ones((2,2), np.uint8)
        rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel_noise) # Remove noise
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel_noise) # Remove noise
        kernel_fill = np.ones((7,7), np.uint8)
        rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel_fill) # Fill holes
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel_fill) # Fill holes
        mask = cv2.bitwise_or(rgb_mask, hsv_mask)
        mask = cv2.resize(mask, (img.shape[1]//2, img.shape[0]//2))

        masks.append(mask)
    return masks
        
    

    

    

    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cable masks from endpoints")
    parser.add_argument(
        "--image",
        type=str,
        default="/home/justinyu/multicable-decluttering/decluttering/DATA_IROS26/objects_in_background/img12.png",
        help="Input image path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/justinyu/multicable-decluttering/decluttering/DATA_IROS26",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="objects_in_background",
        help="Output filename prefix",
    )
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


    vis_img = img_down.copy()

    for pt in endpoints:
        # Scale points back to original size (multiply by 2 since img_down was //2)
        # Ensure coordinates are integers for OpenCV drawing functions
        x = int(pt[0])
        y = int(pt[1])
    
        # Draw a solid red circle
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(vis_img, (y, x), radius=5, color=(255, 0, 0), thickness=-1)
    
        # Optional: Add a small outline to make points pop
        cv2.circle(vis_img, (y, x), radius=6, color=(255, 255, 255), thickness=1)

    # 3. Save the result
    cv2.imwrite(
        f"{args.output_dir}/{args.prefix}_endpoints.png",
        cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
    )

    print(f"Detected {len(endpoints)} endpoints (expected {3})")

    print(endpoints)

    import matplotlib.pyplot as plt

    # masks = get_all_masks(img, endpoints)
    # for idx, pt in enumerate(endpoints):
    #     cv2.imwrite(f"{args.output_dir}/{args.prefix}_mask_{idx}.png", masks[idx])

    masks = []
    for idx, pt in enumerate(endpoints):
        mask = get_mask(img, pt)
        masks.append(mask)
        cv2.imwrite(f"{args.output_dir}/{args.prefix}_mask_{idx}.png", mask)

    # combined_mask_so_far = np.zeros_like(masks[0])

    # for i in range(len(masks)):
    #     # Remove pixels that were already claimed by a previous mask
    #     masks[i] = cv2.bitwise_and(masks[i], cv2.bitwise_not(combined_mask_so_far))
        
    #     # Update the "already claimed" tracker
    #     combined_mask_so_far = cv2.bitwise_or(combined_mask_so_far, masks[i])
    #     cv2.imwrite(f"{args.output_dir}/{args.prefix}_mask_{idx}.png", masks[i])

    # Save 2-row plot: original + all masks

    
    n = len(masks)
    ncols = (n + 2) // 2  # +1 for original image, then split across 2 rows
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10))
    axes = axes.flatten()
    axes[0].imshow(cv2.cvtColor(img_down, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i, mask in enumerate(masks):
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"Mask {i}")
        axes[i + 1].axis("off")
    # Hide any unused axes
    for j in range(n + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{args.prefix}_all_masks.png", dpi=150)
    plt.close()
    print(f"Saved {args.output_dir}/{args.prefix}_all_masks.png")
