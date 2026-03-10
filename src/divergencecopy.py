import cv2
import time
import copy
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from decluttering.src.run_constants import *
from utils.tracer.tusk_pipeline.tracer import TraceEnd


def visualize_multiple_paths(img, path_list, color_list=None, heat_thresh=None, black=False):
    """
    Draws multiple traces for multiple wires on one image.
    If heat_thresh is not specified, will use solid colors.
    If heat_thresh is specified, will plot density heatmap.
    """
    if color_list is None:
        color_list = [None for _ in path_list]
    try:
        if len(path_list) != len(color_list):
            raise ValueError("len(path_list) must == len(color_list).")
    except Exception as e:
        raise ValueError(f"path_list, color_list must be LISTS. \n{e}")
    
    def color_heatmap(color, i, path):
        if color is not None:
            pct = (1 - np.array(color)[i]) / 3
        else:
            pct = i / len(path)
        pct1 = colorsys.hsv_to_rgb(pct, 1, 1)
        return [idx * 255 for idx in pct1][:3]
    
    img = img.copy()
    width = 2 if not black else 5
    
    for path, color in zip(path_list, color_list):
        for i in range(len(path) - 1):
            
            if not isinstance(path, OrderedDict):
                pt1 = tuple(path[i].astype(int))
                pt2 = tuple(path[i + 1].astype(int))
            else:
                path_keys = list(path.keys())
                pt1 = path_keys[i]
                pt2 = path_keys[i + 1]

            if heat_thresh is not None:
                if color[i] > heat_thresh:
                    cv2.line(img, pt1, pt2, color_heatmap(color, i, path), width)
            else:
                cv2.line(img, pt1, pt2, color, width)
    return img


def middle_ones_neighbor_sum(bin_skel: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    neighbor_sum3 = convolve(bin_skel, kernel, mode='constant')
    return np.multiply(neighbor_sum3, bin_skel)


def find_points_of_interest(bin_skel: np.ndarray) -> np.ndarray:
    """
    Given a binary skeleton image containing the combined portion of two wires traces,
    returns all possible points of divergence (not filtered out for global endpoints).
    """
    return np.flip(np.argwhere(middle_ones_neighbor_sum(bin_skel) == 2), axis = 1)


def remove_tri_intersections(bin_skel: np.ndarray) -> np.ndarray:
    """
    Given a binary skeleton image, remove pixels that are part of a 3-way intersection.
    """
    bin_skel[np.where(middle_ones_neighbor_sum(bin_skel) == 4)] = 0
    return bin_skel


def div_points_from_mask(pts_of_interest: np.ndarray, endpts: np.ndarray, 
                         trace_list: np.ndarray, threshold=20) -> np.ndarray:
    """
    Filters out points of interest for divergence if close to global endpoints.
    """
    divergence_pts = np.empty([0, 2])
    for poi in pts_of_interest:
        close_to_endpt = False
        close_to_finish = False

        for glob_endpt in np.flip(endpts, axis = 1):
            if np.linalg.norm(glob_endpt - poi) <= threshold:
                close_to_endpt = True

            # if finished case, don't want to include points close to the last point of the trace
            if np.linalg.norm(np.flip(trace_list[-1]) - poi) <= threshold:
                print(f"Close to finish: {trace_list[-1]} (last point in trace) too close to {poi} (poi)")
                close_to_finish = True

        if not close_to_endpt and not close_to_finish:
            divergence_pts = np.append(divergence_pts, np.array([poi]), axis=0)
    return divergence_pts


def divergence_points(ep_idx_1: int, ep_idx_2: int, trace_list: list,
                      img: np.ndarray, endpts: np.ndarray) -> np.ndarray:
    """
    Given the indices of two starting endpoints with diverging traces and trace list
    for all endpoints returns all points of divergence to perform an IP maneuver on.
    """
    div_points = np.empty([0, 2])
    trace_1, trace_2 = trace_list[ep_idx_1], trace_list[ep_idx_2]
    
    def binary_mask(trace, color):
        dark = np.zeros(img.shape, dtype=np.uint8)
        mask = visualize_multiple_paths(dark, [trace], [color], black=True)

        mask_bin = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        max_val = mask_bin.max()
        if max_val == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)
        mask_bin = (mask_bin // max_val).astype(np.uint8)
        if mask_bin.max() != 1:
            # Defensive fallback for unexpected grayscale values.
            mask_bin = (mask_bin > 0).astype(np.uint8)
        return mask_bin

    mask_1_bin = binary_mask(trace_1, CYAN)
    mask_2_bin = binary_mask(trace_2, MAGENTA)
    
    overlap = np.where(mask_1_bin + mask_2_bin == 2, 1, 0)
    num_labels, labels = cv2.connectedComponents(overlap.astype(np.uint8))
    
    img_num_pixels = img.shape[0] * img.shape[1]
    pct_screen_space = 5e-05    # reject component if less than 0.005% of screen space
    
    if num_labels == 1:
        return None
    else:
        overlap_filtered = None
        overlap_area = []
        
        for i in range(1, num_labels):
            overlap_area.append((labels == i).sum())
        overlap_filtered = np.where(labels == np.argmax(overlap_area) + 1, 1, 0).astype(np.uint8)
        
        if (overlap_filtered == 1).sum() / img_num_pixels < pct_screen_space:
            return None
    
    bin_skel_img = skeletonize(overlap_filtered, method="lee")
    skel_dilated = cv2.dilate(bin_skel_img.astype(np.uint8), np.ones((5,5), np.uint8))
    
    bin_skel_img = skeletonize(skel_dilated, method="lee")
    pts_interest = find_points_of_interest(bin_skel_img.astype(int))

    dp_from_mask = np.array(div_points_from_mask(pts_interest, endpts, trace_list))
    return np.append(div_points, dp_from_mask, axis=0)


def get_all_div_points(img: np.ndarray, endpts: np.ndarray, trace_list=None, endpt_table=None, densities=None, viz=False, name="") -> dict:
    """
    Given the raw image and all endpoints, detects all divergence points / POIs.
    """
    if trace_list is None or endpt_table is None or densities is None:
        trace_list, endpt_table, densities = get_trace_list_colored(img, endpts, viz=viz, name=name)
    
    not_pairs = lambda ep_pair: ep_pair[::-1] not in endpt_table
    yes_pairs = lambda ep_pair: ep_pair[::-1] in endpt_table

    # [[start, end]...]. (end = -1 if trace ended early, may be in corr_endpts.)
    filtered_table = list(filter(not_pairs, endpt_table))
    print(f"Filtered table: {filtered_table}")
    
    corr_endpts = list(filter(yes_pairs, endpt_table))
    set_of_sets = {frozenset(l) for l in corr_endpts}

    asc_endpts  = [list(s) for s in set_of_sets]
    print(f"Associated endpts: {asc_endpts}\n")
    
    for start_idx, end_idx in filtered_table:
        div_points = None
        
        if end_idx == -1:
            print(f"Processing endpt: [{start_idx}, -1]")
            
            # if filtered_table == [[2, -1], [3, 2]], search for endpt 2 (to find 3, the non -1 endpoint)
            found = False
            for filtered_start_idx, filtered_end_idx in filtered_table:
                if filtered_end_idx == start_idx:
                    div_points = divergence_points(start_idx, filtered_start_idx, trace_list, img, endpts)
                    found = True
                    break
            
            # otherwise, search through asc_endpts to find any non -1 endpoints associated with start_idx
            if not found:
                for asc_start_idx, asc_end_idx in asc_endpts:
                    # order of endpoints not preserved; use whichevers != start idx to perform divergence
                    if asc_end_idx == start_idx:
                        div_points = divergence_points(start_idx, asc_start_idx, trace_list, img, endpts)
                        break
                    elif asc_start_idx == start_idx:
                        div_points = divergence_points(start_idx, asc_end_idx, trace_list, img, endpts)
                        break
            
            # last resort
            if div_points is None or len(div_points) == 0:
                print(f"final")

                trace1 = trace_list[start_idx]
                max_corr = 0
                end_idx = -1
                
                for idx, trace2 in enumerate(trace_list):
                    if idx == start_idx:
                        continue
                    overlap = 0
                    for pt in trace1:
                        if pt in trace2:
                            overlap += 1
                    if overlap > max_corr:
                        max_corr = overlap
                        end_idx = idx
                        
                div_points = divergence_points(start_idx, end_idx, trace_list, img, endpts)
        else:
            div_points = divergence_points(start_idx, end_idx, trace_list, img, endpts)
        
        # found a div point
        if div_points is not None and len(div_points) > 0:
            output = {
                "trace_list": trace_list,
                "densities":  densities,
                "div_points": div_points,
                "asc_endpts": asc_endpts,
                "start_idx":  start_idx,
                "end_idx":    end_idx
            }
            print(f"Found {len(div_points)} divergence points")
            return output
        else:
            print(f"No div points found: [{start_idx}, {end_idx}]")
            continue
    
    # No div points found. All asc_endpts should be correlated.
    output = {
        "trace_list": trace_list,
        "densities":  densities,
        "asc_endpts": asc_endpts
    }
    return output
