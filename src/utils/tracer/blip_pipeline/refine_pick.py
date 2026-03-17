import numpy as np
from divergences import *
import cv2
import matplotlib.pyplot as plt
import os

TRACES_DIR = '../data/traces/'

""" Methods for Import"""

def refine_pick_pts(img, trace_t, pick_pts, viz=True, radius=20, num_neighbors=5):
    w, h = img.shape
    xx, yy = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

    best_pts = []
    for pick_pt in pick_pts:
        neighbors_idx = np.argsort(np.linalg.norm(trace_t-pick_pt, axis=1))[:num_neighbors]
        candidates = [pick_pt] + [list(trace_t[idx]) for idx in neighbors_idx]
        best_pt, min_sum = None, np.inf
        for candidate in candidates:
            y, x = candidate
            mask = (xx-x)**2 + (yy-y)**2 < radius**2
            extracted_img = img * mask
            curr_sum = extracted_img.sum()
            if curr_sum < min_sum:
                min_sum = curr_sum
                best_pt = candidate
        best_pts.append(best_pt)
    
    if viz:
        for best_pt, pick_pt in zip(best_pts, pick_pts):
            best_x, best_y = best_pt[1], best_pt[0]
            div_x, div_y = pick_pt[1], pick_pt[0]
            plt.imshow(img)
            plt.scatter(div_x, div_y, c='r', marker='*', label='Original grasp point')
            plt.scatter(best_x, best_y, c='b', marker='o', label='Improved grasp point')
            plt.legend()
            plt.show()

    return best_pts

""" Methods for Testing """

def _refine_pick_for_trace_id(trace_id, pick_pts, viz=True, radius=20, num_neighbors=5):
    
    # get the original (i.e. without noise) trace
    if not os.path.isfile(TRACES_DIR + f'{trace_id}/trace_0.npy'):
        raise Exception('Original trace file does not exist!')
    trace_t = np.load(TRACES_DIR + f'{trace_id}/trace_0.npy', allow_pickle=True)[0]

    if not os.path.isfile(TRACES_DIR + f'{trace_id}/img_0.png'):
        raise Exception('Original image does not exist!')
    img = cv2.imread(TRACES_DIR + f'{trace_id}/img_0.png', cv2.IMREAD_GRAYSCALE)
    w, h = img.shape
    xx, yy = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

    best_pts = []
    for pick_pt in pick_pts:
        neighbors_idx = np.argsort(np.linalg.norm(trace_t-pick_pt, axis=1))[:num_neighbors]
        candidates = [pick_pt] + [list(trace_t[idx]) for idx in neighbors_idx]
        best_pt, min_sum = None, np.inf
        for candidate in candidates:
            y, x = candidate
            mask = (xx-x)**2 + (yy-y)**2 < radius**2
            extracted_img = img * mask
            curr_sum = extracted_img.sum()
            if curr_sum < min_sum:
                min_sum = curr_sum
                best_pt = candidate
        best_pts.append(best_pt)
    
    if viz:
        for best_pt, pick_pt in zip(best_pts, pick_pts):
            best_x, best_y = best_pt[1], best_pt[0]
            div_x, div_y = pick_pt[1], pick_pt[0]
            plt.imshow(img)
            plt.scatter(div_x, div_y, c='r', marker='*', label='Original grasp point')
            plt.scatter(best_x, best_y, c='b', marker='o', label='Improved grasp point')
            plt.legend()
            plt.show()

    return best_pts


if __name__ == "__main__":
    if not os.path.exists(TRACES_DIR):
        raise Exception('Traces directory does not exist!')
    for trace_id in os.listdir(TRACES_DIR):
        if os.path.isdir(TRACES_DIR + f'/{trace_id}'):
            trace_t, noisy_traces = load_traces_for_trace_id(trace_id)
            divergence_pts = get_divergence_pts(trace_t=trace_t, noisy_traces=noisy_traces)
            best_pts = _refine_pick_for_trace_id(trace_id=trace_id, pick_pts=divergence_pts)
