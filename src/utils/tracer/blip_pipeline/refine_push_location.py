import numpy as np
from decluttering.data.tracer.blip_pipeline.divergences import *
import cv2
import matplotlib.pyplot as plt
import os
import time
TRACES_DIR = '../data/traces/'

""" Methods for Import"""

def find_crossings(img, points, viz=True, radius=20, num_neighbors=1):
    start_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    w, h = img.shape
    xx, yy = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

    # best_pts = []
    # max_sum = 0
    max_sums = []
    for curr_pt in points:
        # get the closest pt
        y, x = curr_pt
        mask = (xx-x)**2 + (yy-y)**2 < radius**2
        extracted_img = img * mask
        curr_sum = extracted_img.sum()
        max_sums.append(curr_sum)
    print(f"Density computation time: {time.time() - start_time}")
    return max_sums

# def find_crossings(img, points, viz=True, radius=20, num_neighbors=1):
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     w, h = img.shape
#     xx, yy = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

#     # best_pts = []
#     # max_sum = 0
#     max_sums = []
#     for i, curr_pt in enumerate(points):
#         # get the closest pt
#         trace_t = [points[j] for j in range(len(points)) if j != i]
#         nearest_pt_idx = np.argsort(np.linalg.norm(trace_t-curr_pt, axis=1))[0]
#         neighbors_idx = [idx for idx in range(max(nearest_pt_idx-num_neighbors, 0), nearest_pt_idx)]
#         candidates = [trace_t[nearest_pt_idx]] + [list(trace_t[idx]) for idx in neighbors_idx]
#         max_sum = 0
#         # best_pt, max_sum = None, 0
#         for candidate in candidates:
#             y, x = candidate
#             mask = (xx-x)**2 + (yy-y)**2 < radius**2
#             extracted_img = img * mask
#             curr_sum = extracted_img.sum()
#             if curr_sum > max_sum:
#                 max_sum = curr_sum
#         max_sums.append(max_sum)

#     return max_sums

'''def refine_push_location_pts(img, trace_t, divergence_pts, viz=True, radius=20, num_neighbors=5):
    w, h = img.shape
    xx, yy = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

    # best_pts = []
    # max_sum = 0
    sums = []
    for divergence_pt in divergence_pts:
        # get the closest pt
        nearest_pt_idx = np.argsort(np.linalg.norm(trace_t-divergence_pt, axis=1))[0]
        neighbors_idx = [idx for idx in range(max(nearest_pt_idx-num_neighbors, 0), nearest_pt_idx)]
        candidates = [trace_t[nearest_pt_idx]] + [list(trace_t[idx]) for idx in neighbors_idx]
        max_sum = 0
        # best_pt, max_sum = None, 0
        for candidate in candidates:
            y, x = candidate
            mask = (xx-x)**2 + (yy-y)**2 < radius**2
            extracted_img = img * mask
            curr_sum = extracted_img.sum()
            if curr_sum > max_sum:
                max_sum = curr_sum
            #     best_pt = candidate
        sums.append(max_sum)
        # best_pts.append(best_pt)
    
    if viz:
        for best_pt, divergence_pt in zip(best_pts, divergence_pts):
            best_x, best_y = best_pt[1], best_pt[0]
            div_x, div_y = divergence_pt[1], divergence_pt[0]
            plt.imshow(img)
            plt.scatter(div_x, div_y, c='r', marker='*', label='Original push point')
            plt.scatter(best_x, best_y, c='b', marker='o', label='Improved push point')
            plt.legend()
            plt.show()

    return sums
    # return best_pts'''

""" Methods for Testing """

def _refine_push_location_pts(trace_id, divergence_pts, viz=True, radius=20, num_neighbors=10):
    
    # get the original (i.e. without noise) trace
    if not os.path.isfile(TRACES_DIR + f'{trace_id}/trace_0.npy'):
        raise Exception('Original trace file does not exist!')
    trace_t = np.load(TRACES_DIR + f'{trace_id}/trace_0.npy', allow_pickle=True)

    if not os.path.isfile(TRACES_DIR + f'{trace_id}/img_0.png'):
        raise Exception('Original image does not exist!')
    img = cv2.imread(TRACES_DIR + f'{trace_id}/img_0.png', cv2.IMREAD_GRAYSCALE)
    w, h = img.shape
    xx, yy = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

    best_pts = []
    for divergence_pt in divergence_pts:
        # get the closest pt
        nearest_pt_idx = np.argsort(np.linalg.norm(trace_t-divergence_pt, axis=1))[0]
        neighbors_idx = [idx for idx in range(max(nearest_pt_idx-num_neighbors, 0), nearest_pt_idx)]
        candidates = [trace_t[nearest_pt_idx]] + [list(trace_t[idx]) for idx in neighbors_idx]
        best_pt, max_sum = None, 0
        for candidate in candidates:
            y, x = candidate
            mask = (xx-x)**2 + (yy-y)**2 < radius**2
            extracted_img = img * mask
            curr_sum = extracted_img.sum()
            if curr_sum > max_sum:
                max_sum = curr_sum
                best_pt = candidate
        best_pts.append(best_pt)
    
    if viz:
        for best_pt, divergence_pt in zip(best_pts, divergence_pts):
            best_x, best_y = best_pt[1], best_pt[0]
            div_x, div_y = divergence_pt[1], divergence_pt[0]
            plt.imshow(img)
            plt.scatter(div_x, div_y, c='r', marker='*', label='Original push point')
            plt.scatter(best_x, best_y, c='b', marker='o', label='Improved push point')
            plt.legend()
            plt.show()

    return best_pts


if __name__ == "__main__":
    if not os.path.exists(TRACES_DIR):
        raise Exception('Traces directory does not exist!')
    print("len of dir: ", os.listdir(TRACES_DIR))
    for trace_id in os.listdir(TRACES_DIR):
            trace_t, noisy_traces = load_traces_for_trace_id(trace_id)
            divergence_pts = get_divergence_pts(trace_t=trace_t, noisy_traces=noisy_traces)
            best_pts = _refine_push_location_pts(trace_id=trace_id, divergence_pts=divergence_pts)
