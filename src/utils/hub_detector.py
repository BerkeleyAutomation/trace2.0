import os

import cv2
from matplotlib import pyplot as plt
import numpy as np


def detect_hubs(endpoints, trace_list, n_hubs=2, offset_px=60, cluster_threshold=150, direction_cos_threshold=0.0):
    if len(endpoints) < 2:
        print("[detect_hubs] Not enough endpoints to detect hubs")
        return []
    
    pts = np.array(endpoints, dtype=float)

    clusters = [[0]]
    for i in range(1, len(pts)):
        centroids = [np.mean(pts[c], axis=0) for c in clusters]
        dists = [np.linalg.norm(pts[i] - c) for c in centroids]
        nearest = np.argmin(dists)
        if dists[nearest] < cluster_threshold or len(clusters) >= n_hubs:
            clusters[nearest].append(i)
        else:
            clusters.append([i])

    print(f"[detect_hubs] Found {len(clusters)} clusters: {[len(c) for c in clusters]} endpoints each")

    press_locations = []
    for cluster_indices in clusters:
        centroid = np.mean(pts[cluster_indices], axis=0)
    
        directions = []
        for ep_idx in cluster_indices:
            trace = trace_list[ep_idx]
            if len(trace) < 5:
                print(f"[detect_hubs] Trace {ep_idx} too short, skipping")
                continue
            # trace[0] is at endpoint, pointing away from hub
            trace_dir = trace[4] - trace[0]  # (x, y) = (col, row)
            norm = np.linalg.norm(trace_dir)
            if norm > 0:
                directions.append(trace_dir / norm)
        
        if len(directions) == 0:
            print(f"[detect_hubs] No valid trace directions for cluster, using centroid as press location")
            press_locations.append(centroid)
            continue

        directions = np.array(directions)
        if len(directions) > 1:
            median_dir = directions[len(directions) // 2]
            cos_sims = directions @ median_dir
            valid = directions[cos_sims > direction_cos_threshold]
            if len(valid) == 0:
                print(f"[detect_hubs] All directions filtered as outliers, using all")
                valid = directions
        else:
            valid = directions
        
        avg_dir_xy = np.mean(valid, axis=0)
        avg_dir_xy /= np.linalg.norm(avg_dir_xy)

        inward_dir_xy = -avg_dir_xy

        centroid_xy = np.array([centroid[1], centroid[0]])
        press_xy = centroid_xy + inward_dir_xy * offset_px
        press_rc = np.array([press_xy[1], press_xy[0]])

        press_locations.append(press_rc)
    return press_locations

def save_hub_press_locations_image(img, press_locations, output_dir, viz=False):
    """Draw hub press locations on the image and save."""
    vis = img.copy()
    
    for i, press_rc in enumerate(press_locations):
        r, c = int(press_rc[0]), int(press_rc[1])
        cv2.drawMarker(vis, (c, r), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        cv2.circle(vis, (c, r), 15, (0, 255, 0), 2)
        cv2.putText(vis, f"Hub {i}", (c + 20, r), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    output_path = os.path.join(output_dir, 'hub_press_locations.png')
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")
    
    if viz:
        plt.figure(figsize=(16, 12))
        plt.imshow(vis)
        plt.title('Hub Press Locations')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
