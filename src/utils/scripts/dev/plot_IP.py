import math
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datetime import datetime

import torch
from decluttering.data.utils.detic_dataloader import DeticDataloader
from decluttering.data.utils.tracer_dataloader import TracerDataloader
from decluttering.src.main_backup import hungarian_matching
# from decluttering.src.run_constants import *
# from decluttering.src.motion_jacobi import *
from decluttering.data.utils.endpoint_detector_dataloader import EndpointDataloader
from decluttering.src.knot_regions import find_open_knot_regions, classify_point_in_knot, plot
from decluttering.src.divergencecopy import *
from decluttering.src.calc_arc_length import get_circle, in_circle, view_circles, get_new_start_pt
import itertools
from pathlib import Path
from skimage.filters import frangi
from skimage.morphology import skeletonize

# Set VIZ = True for matplotlib plots, False to hide
VIZ = True

BIMANUAL_DECLUTTER = True

# CHANGE THIS ACCORDING TO NUMBER OF ENDPTS IN SCENE
NUM_ENDPTS = 8

if NUM_ENDPTS == 4: #TIER 1 = 2 tangentials, 4 endpoints, TIER 2 = 3-4 tangentials, 4 endpoints
    TIER = 2
elif NUM_ENDPTS == 6: #TIER 3 = 3-4 tangentials, 6 endpoints
    TIER = 3
elif NUM_ENDPTS == 8: #TIER 4 = 4-5 tangentials, 8 endpoints
    TIER = 4
else:
    raise ValueError(NUM_ENDPTS)

SAVE_DIR = '/home/justinyu/multicable-decluttering/decluttering/scripts/dev/outputs/DATA_PLOTS'

curr_time = datetime.now()
txt_path  = f"{SAVE_DIR}/tier_{TIER}/{curr_time}/metrics.txt"
os.makedirs(f"{SAVE_DIR}/tier_{TIER}/{curr_time}/", exist_ok=True)

TRACE_FINISHED = False

def plot(dir, viz, name):
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{name}_{dir}.png")
    if viz:
        plt.show()
    else:
        plt.close()

# gets the bounding box of an object from a binary mask
def get_oriented_bounding_box(binary_mask):
    """
    Fit the tightest oriented bounding box around a 2D binary mask.
    
    Parameters:
    binary_mask (numpy.ndarray): A 2D numpy array representing the binary mask image.
    
    Returns:
    tuple: The (center (x, y), (width, height), angle of rotation) of the bounding box.
    """
    # Ensure the input is a binary mask
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour, assuming the mask is connected
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area bounding rectangle
    rect = cv2.minAreaRect(largest_contour)
    
    angle = rect[2]
    
    if rect[1][0] > rect[1][1]:
        angle = angle + 90
    if np.cos(np.radians(angle)) > 0:
        angle = angle + 180
        
    return rect, angle, largest_contour

# gets the plot mask based don the binary mask and the bounding box
def plot_mask_with_bbox(binary_mask, rect, contour, plt_path):
    """
    Plot the binary mask with the oriented bounding box.
    
    Parameters:
    binary_mask (numpy.ndarray): A 2D numpy array representing the binary mask image.
    rect (tuple): The (center (x, y), (width, height), angle of rotation) of the bounding box.
    contour (numpy.ndarray): Contour of the largest component in the binary mask.
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    angle = rect[2]
    if rect[1][0] > rect[1][1]:
        angle = angle + 90
        
    if np.cos(np.radians(angle)) > 0:
        angle = angle + 180
    
    plt.figure()
    plt.imshow(binary_mask, cmap='YlOrRd')
    # plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
    plt.plot(*zip(*np.vstack([box, box[0]])), 'b-', linewidth=2)
    plt.scatter(rect[0][0], rect[0][1], color='green', s=100)
    plt.plot([rect[0][0], rect[0][0]-50*np.sin(np.radians(angle))*5], [rect[0][1], rect[0][1]+50*np.cos(np.radians(angle))*5], linewidth=2)
    plt.title("Binary Mask with Oriented Bounding Box")
    
    # grasp_vec = [-np.sin(np.radians(angle)), np.cos(np.radians(angle))] 
    # print("Grasp Vec: ", grasp_vec)
    plt.savefig(f"{SAVE_DIR}/{plt_path}_mask_with_bbox.png")
    
    plt.show()

import torch.nn.functional as F
def dilate_masks(mask_tensor, kernel_size=3):
    # Add batch and channel dimensions if needed
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    
    # Convert boolean to float
    mask_tensor = mask_tensor.float()
    
    # Create kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask_tensor.device)
    
    # Dilate using max_pool2d
    dilated = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
    
    # Convert back to boolean if input was boolean
    if mask_tensor.dtype == torch.bool:
        dilated = dilated > 0
    
    # Remove extra dimensions if they were added
    dilated = dilated.squeeze()
    return dilated

def get_detic_masks(img, detic):
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

#NEVER USED --> REMOVE
def trace_and_single_declutter(img, trace_list, detic):
    detic.create()
    detic.default_vocab()
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
    for trace in trace_list:
        for x, y in trace:
            mask_num = detic_masked_arr[y*2][x*2]
            mask_ind = mask_num - 1
            is_intersecting_mask = bool_detic_masked_arr[y*2][x*2]
            if is_intersecting_mask:
                # declutter
                bin_mask = out['masks'][mask_ind].squeeze().cpu().numpy()
                rect, contour = get_oriented_bounding_box(bin_mask)
                plot_mask_with_bbox(bin_mask, rect, contour)
                # plt.show()
                print(rect)
                center = np.array(rect[0])
                angle = rect[2]
                if rect[1][0] > rect[1][1]:
                    angle = angle + 90
                if np.cos(np.radians(angle)) > 0:
                    angle = angle + 180
                ## BIN DROP:
                # interface.home()
                # interface.open_grippers()
                # perform_point_grasp_and_bin_drop_pipeline(center, angle, interface)
    
# Force BRIO camera to take new pic, then resize
def predict_pts(img, endpt_model, bool_detic_mask = None, detic_mask = None, plt_path = ""):
    
    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)) # (1060, 1904, 3)
    endpoints = endpt_model.predict(img_down)['endpoints']
    
    if bool_detic_mask is not None:
        # Filter out endpoints that are within a mask
        filtered_endpoints = []
        for endpt in endpoints:
            if bool_detic_mask[endpt[0]*2][endpt[1]*2] == 0:
                filtered_endpoints.append(endpt)
    
    print(f"Detected {len(endpoints)} endpoints")
    
    plt.figure(figsize=(16, 12))
    plt.imshow(img_down)
    if detic_mask is not None:
        detic_mask_down = cv2.resize(detic_mask.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy().astype('uint8')*255, (img.shape[1]//2, img.shape[0]//2))
        plt.imshow(detic_mask_down, alpha=0.6)
    
    for endpt in endpoints:
        plt.scatter(endpt[1], endpt[0], c="blue", s=100)
    for endpt in filtered_endpoints:
        plt.scatter(endpt[1], endpt[0], c="red", s=80)
    plt.show()
    # plt.savefig(f"{SAVE_DIR}/{plt_path}_endpts_detic.png")
        
    return filtered_endpoints


def count_circles(trace, click = False, viz = False, img = None):
    circles = []
    start_pt = trace[0]
    
    next_idx = 1
    while next_idx != trace.shape[0]-1:
        next_pt = trace[next_idx]
        circle, new_next_idx = get_circle(start_pt, next_pt, next_idx, trace)
        new_start_pt = get_new_start_pt(trace, new_next_idx, circle._center, circle.radius)
        start_pt = new_start_pt
        next_idx = new_next_idx
        circles.append(circle)
    
    if not in_circle(trace[-1], circles[-1]):
        radius = circles[-1].radius
        start, next = start_pt, trace[next_idx]
        theta = np.arctan2((next[1] - start[1]), (next[0] - start[0]))
        center_x = start[0] + np.cos(theta) * radius
        center_y = start[1] + np.sin(theta) * radius
        circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
        circles.append(circle)

    if viz:
        if click:
            return view_circles(trace, circles, click=True, img = img)
        else:
            return view_circles(trace, circles, img = img)
    
    print(len(circles))
    return len(circles)

def div_point_ips(interface, output, img_down, img, plt_path):
    if "div_points" not in output.keys():
        print("\nNO DIV_PTS FOUND.")

        trace_list = output['trace_list']
        asc_endpts = output['asc_endpts']
        plot_points = []
        
        for indices in asc_endpts:
            endpt1, endpt2 = indices
            trace1, trace2 = trace_list[endpt1], trace_list[endpt2]
            start, end = trace1[0], trace2[0]
            plot_points.append((start, end))
        plt.figure(figsize=(16, 12))
        plt.imshow(img_down)
        
        for coord, color in zip(plot_points, COLORS[:len(plot_points)]):
            # Convert RGB color to plt format (0->1)
            normal = tuple(c / 255.0 for c in color)
            x0, y0 = coord[0]
            x1, y1 = coord[1]
            plt.scatter([x0, x1], [y0, y1], color=normal, s=100)
        
        plt.tight_layout()
        plt.title(f"Result: {len(plot_points*2)} / {NUM_ENDPTS} endpoints matched")
        plt.savefig(f"{SAVE_DIR}/{plt_path}_final_plot.png")
        plt.show()
        TRACE_FINISHED = True
        return False
        # break

    # if div_points exists: go resolve it
    else:
        trace_list  = output["trace_list"]
        density_nrm = output["densities"]
        div_points  = output["div_points"]
        asc_endpts  = output["asc_endpts"]
        endpt_1     = output["start_idx"]
        endpt_2     = output["end_idx"]
        
        densities_pad = [np.concatenate((np.zeros(4), d)) for d in density_nrm]

        ###plt###
        # fig, ax = plt.subplots(figsize=(16, 12))
        # ax.imshow(img_down)
        # for div_pt in div_points:
        #     circle = patches.Circle(div_pt, radius=25, linewidth=2, 
        #                             edgecolor="lime", facecolor="none")
        #     ax.add_patch(circle)
        
        # fig.suptitle(f"Detected {len(div_points)} divergence points (green)")
        # plot("div_points", False, plt_path)
        ########
        
        do_push = True
        # for div_pt in div_points:
        #     print("\nClassifying (point in knot?)...")
            
        #     dense_count = classify_point_in_knot(img_down, div_pt, trace_list,
        #                               densities_pad, viz=VIZ, name=plt_path)
            
        #     if dense_count > 0:
        #         centroid, cent_area = find_open_knot_regions(
        #             img_down, div_pt, viz=VIZ, name=plt_path)
                
        #         if centroid is not None:
        #             arm = goto_gripper_px(centroid * 2, interface)
        #             time.sleep(0.5)

        #             interface.open_grippers()
        #             rotate_gripper(arm, interface)
                    
        #             interface.home()
        #             interface.close_grippers()

        #             with open(txt_path, "a") as f:
        #                 f.write(f"Cluster Dilation: {dense_count} Dense px, Centroid Area = {cent_area:.2f} px, Distance to Div_pt = {math.dist(centroid, div_pt):.2f} px\n")
        #             do_push = False
        #             return True
        #########


        if do_push:
            print("Planning Push IP Move...")
            end_coord, start_coord, vec_angle, vec_dist = get_push_coords(div_points, img_down,
                                        trace_list, endpt_1, endpt_2, viz=False, name=plt_path)
            print("DONE")
            end_coord   *= 2    # Scale to original size
            start_coord *= 2
            
            vector_0 = (end_coord[0] - start_coord[0]) * VECTOR_SCALE
            vector_1 = (end_coord[1] - start_coord[1]) * VECTOR_SCALE
            
            new_vec0 = start_coord[0] + vector_0
            new_vec1 = start_coord[1] + vector_1
            
            ###plt###
            plt.figure(figsize=(16, 12))
            plt.imshow(img)     # Vec need original size
            plt.title(f"Distance <Start Coord -> Div_pt Coord> = {vec_dist:.2f}")

            plt.arrow(start_coord[0], start_coord[1], vector_0, vector_1,
                    width=2, head_width=20, head_length=40, fc='r', ec='r')
            plot("vec_plan", VIZ, plt_path)
            #########
            
            # try:
            #     print("Executing Push IP Move...")
            #     perform_push_through([new_vec0, new_vec1], start_coord, interface)
            #     with open(txt_path, "a") as f:
            #         f.write(f"Push Through: Angle to Bisector = {vec_angle:.2f} deg, Distance to Div_pt = {vec_dist:.2f} px\n")
            #     print("DONE")
            #     return True
            # except Exception as e:
            #     print(f"\n{e}\n Motion planning failure. Retry with new IP plan.")
            #     return False



def get_closest_trace_idx(poi, trace):
    return np.argmin(np.linalg.norm(np.array(trace) - np.array(poi)[None, ...], axis=1))

def vector_angle(vec1, vec2):
    return np.degrees(np.arccos(np.dot(np.array(vec1), np.array(vec2))))

def vector_distance(vec1, vec2, axis=None):
    return np.linalg.norm(np.array(vec1) - np.array(vec2), axis=axis)

def vector_direction(vec1, vec2):
    return (np.array(vec1) - np.array(vec2)) / vector_distance(vec1, vec2)


def get_push_coords(poi_coords, img_rgb, trace_list, endpt_1, endpt_2,
                    keepout_dist=KEEPOUT_DIST, viz=False, name=""):
    """
    Algorithm for IP move planning.
    - Output: (poi_coord, start_coord)
    - Inputs:
        - poi_coords: array(shape=(n, 2)), {n: # of points of interest, 2: (x, y)}
        - img_rgb: array(shape=(h, w, 3)), {h: height=1060, w: width=1904}
        - trace_list: list(array(shape=(m, 2)), {m: # of trace points,  2: (x, y)}
       
        - endpt_1/2: int(endpoint indices)
        - keepout_dist: decrease this to push IP Vectors in closer
        - viz: Set this True for matplotlib visuals, False to hide
    """
    
    # array(n, 2),  list
    trace_1 = trace_list[endpt_1]
    trace_2 = trace_list[endpt_2]
    
    # coord(y, x),   list
    for p, poi_coord in enumerate(poi_coords):
        plt.close()

        # integers
        close_idx_1 = get_closest_trace_idx(poi_coord, trace_1)
        close_idx_2 = get_closest_trace_idx(poi_coord, trace_2)

        # coord(x, y)
        poi_trace_1 = trace_1[close_idx_1]
        poi_off_1   = trace_1[close_idx_1 - 1]
        poi_trace_2 = trace_2[close_idx_2]
        poi_off_2   = trace_2[close_idx_2 - 1]

        # Black image
        mask = np.zeros(img_rgb.shape, dtype=np.uint8)
        for trace in trace_list:
            mask = visualize_multiple_paths(
                mask, [trace], [WHITE], black=True)
        
        # Direction vectors
        poi_vec_1 = vector_direction(poi_trace_1, poi_off_1)
        poi_vec_2 = vector_direction(poi_trace_2, poi_off_2)
        
        # integer
        vec_angle = vector_angle(poi_vec_1, poi_vec_2)
        if vec_angle > 90:
            poi_vec_2 = -poi_vec_2  # force it to be acute

            vec_angle = vector_angle(poi_vec_1, poi_vec_2)
            if vec_angle > 90:
                import pdb; pdb.set_trace()
        
        poi_vec_bisect = (poi_vec_1 + poi_vec_2) / 2 # Direction
        bi_point = (poi_trace_1 + poi_trace_2) / 2   # coord(x, y)
        
        # Save and plot
        plt.figure(figsize=(16, 12))
        plt.imshow(img_rgb)
        plt.tight_layout()
        plt.title(f"ANGLE BETWEEN BLUE TANGENTS: {vec_angle:.2f}")
        plt.axline(poi_trace_1, poi_off_1)
        plt.axline(poi_trace_2, poi_off_2)
        plt.axline(bi_point, bi_point + poi_vec_bisect, color="r")
        
        plt.savefig(f"{SAVE_DIR}/{name}_vec_angle.png")
        if viz:
            plt.show() 
        plt.close()
        
        dilate_kernel4 = np.ones((4, 4), np.uint8)
        dilated_binary_img = ~cv2.dilate(mask[:,:,0], dilate_kernel4)
        dilated_binary_img[dilated_binary_img != 0] = 1
        
        # bbox = (550//2, 1300//2, 1900//2, 2650//2)
        
        bbox = (220, 510, 780, 1070)
        
        plt.clf()
        plt.imshow(dilated_binary_img[bbox[0]:bbox[1], bbox[2]:bbox[3]], cmap='gray')
        plt.axis('off')
        plt.savefig(f"{SAVE_DIR}/{name}_dilated_binary_img.png", bbox_inches='tight', pad_inches=0)
        
        _, labels, _,_ = cv2.connectedComponentsWithStats(dilated_binary_img)
        dist_transform = cv2.distanceTransform(dilated_binary_img, cv2.DIST_L2, 5)
        
        plt.imshow(np.clip(dist_transform, 0, dist_transform.max()*7/16)[bbox[0]:bbox[1], bbox[2]:bbox[3]])
        # plt.imshow(dist_transform)

        plt.axis('off')
        plt.savefig(f"{SAVE_DIR}/{name}_dist_transform_img.png", bbox_inches='tight', pad_inches=0)
        
        RIDGE_THRESH = 0.0025
        ridges = frangi(dist_transform, black_ridges=False, sigmas=range(1, 5))
        ridges[ridges <  RIDGE_THRESH] = 0
        ridges[ridges >= RIDGE_THRESH] = 1
        # ridges[dist_transform < 0.1] = 0
        
        skel_ridges = skeletonize(ridges, method="lee")
        # get rid of ridges within 1% of the image edge
        rid0 = int(0.01 * img_rgb.shape[0])
        rid1 = int(0.01 * img_rgb.shape[1])

        skel_ridges[:rid0, : ] = 0
        skel_ridges[-rid0:, :] = 0
        skel_ridges[:, :rid1 ] = 0
        skel_ridges[:, -rid1:] = 0
        
        plt.imshow(skel_ridges[bbox[0]:bbox[1], bbox[2]:bbox[3]], cmap='gray')
        plt.axis('off')
        plt.savefig(f"{SAVE_DIR}/{name}_ridges_img.png", bbox_inches='tight', pad_inches=0)
        
    
        ridge_points = find_points_of_interest(skel_ridges.astype(int) // 255)
        # sort ridge_points by distance to the poi_coord
        ridge_points = np.array(sorted(ridge_points, key=lambda x: vector_distance(x, poi_coord)))
        
        # remove pixels at the intersection of 3 ridges
        skel_ridges  = remove_tri_intersections(skel_ridges)
        
        ITER_THRESH = 2   # number of ridge points to check before moving on to next poi_coord
        ANGLE_THRESH = 25 # max angle between IP plan and bisect vector to be considered valid
        DIST_THRESH = 200 # max distance between planned start coordinate and divergence point

        # go through ridge_points IN ORDER until thresh
        for i, closest_point in enumerate(ridge_points):
            plt.close()

            if i > ITER_THRESH and p < len(poi_coords) - 1:
                print(f"Checked {ITER_THRESH} ridge_points, moving to next poi_coord")
                break

            plt.figure(figsize=(16, 12))
            plt.imshow(img_rgb)
            plt.imshow(skel_ridges, alpha=0.3)
            plt.scatter(ridge_points[:, 0], ridge_points[:, 1], s=10, c="r", zorder=10)
            plt.scatter(closest_point[0], closest_point[1], s=20, c="g", zorder=10)
            plt.scatter(poi_coord[0], poi_coord[1], s=40, c="b", zorder=10)
            plt.tight_layout()
            
            dilate_kernel5 = np.ones((5, 5), np.uint8)
            dilated_skel_ridges = cv2.dilate(skel_ridges.astype(np.uint8), dilate_kernel5)
            
            _, labels, _,_ = cv2.connectedComponentsWithStats(dilated_skel_ridges // 255)
            closest_index  = labels[closest_point[1], closest_point[0]]
            closest_ridge  = np.where(labels == closest_index, 1, 0).astype(np.uint8)
            
            closest_ridge[dist_transform < keepout_dist] = 0
            skel_keepout   = skeletonize(closest_ridge, method="lee")
            
            # points after keepout distance is applied
            keepout_points = find_points_of_interest(skel_keepout.astype(int))
            
            if keepout_points is None or len(keepout_points) == 0:
                print("No keepout_points, moving to next ridge_point")
                continue
            
            # finds the closest point to the poi_coord
            start_coord = keepout_points[np.argmin(vector_distance(keepout_points, poi_coord, axis=1))]
            plt.scatter(start_coord[0], start_coord[1], s=80, c="m", zorder=10)
            
            poi_vec_start  = vector_direction(poi_coord, start_coord)
            poi_vec_bisect = (poi_vec_1 + poi_vec_2) / 2
            
            # (start-poi)coord vector
            plt.axline(start_coord, slope=poi_vec_start[1] / poi_vec_start[0], color='c')
            # Bisect direction vector
            plt.axline(poi_coord, slope=poi_vec_bisect[1] / poi_vec_bisect[0], color='r')

            vec_angle = vector_angle(poi_vec_start, poi_vec_bisect)
            if vec_angle >= ANGLE_THRESH:
                poi_vec_bisect = -poi_vec_bisect

                vec_angle = vector_angle(poi_vec_start, poi_vec_bisect)
                if (i <= ITER_THRESH and vec_angle >= ANGLE_THRESH) or vec_angle >= ANGLE_THRESH * 2:
                    print(f"vec_angle {vec_angle:.2f} >= {ANGLE_THRESH}, moving to next ridge_point")
                    continue
            
            vec_dist = vector_distance(start_coord, poi_coord)
            if i <= ITER_THRESH and vec_dist >= DIST_THRESH:
                print(f"(start-poi)coord >= {DIST_THRESH}, moving to next ridge_point")
                continue

            plt.title(f"Angle <Planned (cyan) -> Bisect (red)> = {vec_angle:.2f}")
            plt.close()
            
            return poi_coord, start_coord, vec_angle, vec_dist
        print("MOVING TO NEXT POI_COORD")




def ip_moves_plot(output, img_down, img, plt_path):
    # if div_points exists: go resolve it
    trace_list  = output["trace_list"]
    density_nrm = output["densities"]
    div_points  = output["div_points"]
    asc_endpts  = output["asc_endpts"]
    endpt_1     = output["start_idx"]
    endpt_2     = output["end_idx"]
    
    densities_pad = [np.concatenate((np.zeros(4), d)) for d in density_nrm]

    ###plt###
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img_down)
    for div_pt in div_points:
        circle = patches.Circle(div_pt, radius=25, linewidth=2, 
                                edgecolor="lime", facecolor="none")
        ax.add_patch(circle)
    
    fig.suptitle(f"Detected {len(div_points)} divergence points (green)")
    # plot("div_points", False, plt_path)
    ########
    
    do_push = True
    for div_pt in div_points:
        print("\nClassifying (point in knot?)...")
        
        dense_count = classify_point_in_knot(img_down, div_pt, trace_list,
                                    densities_pad, viz=True, name=plt_path)
        
        if dense_count > 0:
            centroid, cent_area = find_open_knot_regions(
                img_down, div_pt, viz=True, name=plt_path)
            
            
            # if centroid is not None:
            #     arm = goto_gripper_px(centroid * 2, interface)
            #     time.sleep(0.5)

            #     interface.open_grippers()
            #     rotate_gripper(arm, interface)
                
            #     interface.home()
            #     interface.close_grippers()

            #     with open(txt_path, "a") as f:
            #         f.write(f"Cluster Dilation: {dense_count} Dense px, Centroid Area = {cent_area:.2f} px, Distance to Div_pt = {math.dist(centroid, div_pt):.2f} px\n")
            #     do_push = False
            #     return True
    #########


    if do_push:
        print("Planning Push IP Move...")
        end_coord, start_coord, vec_angle, vec_dist = get_push_coords(div_points, img_down,
                                    trace_list, endpt_1, endpt_2, viz=False, name=plt_path)
        print("DONE")
        end_coord   *= 2    # Scale to original size
        start_coord *= 2
        
        vector_0 = (end_coord[0] - start_coord[0]) * VECTOR_SCALE
        vector_1 = (end_coord[1] - start_coord[1]) * VECTOR_SCALE
        
        new_vec0 = start_coord[0] + vector_0
        new_vec1 = start_coord[1] + vector_1
        
        ###plt###
        bbox = (550, 1300, 1900, 2650)

        plt.figure(figsize=(16, 12))
        plt.imshow(img[bbox[0]:bbox[1], bbox[2]:bbox[3]])     # Vec need original size
        # plt.title(f"Distance <Start Coord -> Div_pt Coord> = {vec_dist:.2f}")

        plt.arrow(start_coord[0] - bbox[2] , start_coord[1] - bbox[0], vector_0, vector_1,
                width=8, head_width=45, head_length=60, fc='b', ec='b')
        plt.axis('off')
        plt.savefig(f"{SAVE_DIR}/{plt_path}_{dir}.png", bbox_inches='tight', pad_inches=0)
        # plot("vec_plan", VIZ, plt_path)
            
def middle_of_box(out):
    return [((out['boxes'][i][0] + out['boxes'][i][2])/2, (out['boxes'][i][1] + out['boxes'][i][3])/2) for i in range(out['boxes'].shape[0])]

def bimanual_filter_grasps(mask_num, out):
    assert isinstance(mask_num, np.ndarray), f"mask_num should be a numpy array but is instead: {type(mask_num)}"
    
    # No masks were hit on any trace, return -1
    if (mask_num.all() == -1):
        return -1
    
    masks = np.delete(mask_num, np.where(mask_num == -1))
    
    masks_set = set(masks)
    
    # If only one mask is left and !=-1, return it for single arm grasping
    if len(masks_set) == 1:
        return [int(list(masks_set)[0])]

    # Iterate through all combinations of pairs of masks to determine bimanual grasp validity; return first valid pair
    for pair in itertools.combinations(masks_set, 2):
        valid, flip_lr =  check_valid_bimanual_grasp(pair, out)
        if valid:
            if flip_lr:
                print(f"Flipping left and right grasp for pair {pair}")
                pair = [pair[1], pair[0]]
            return list(pair)
        
    # If no valid pair is found, return any single mask index (continue to single arm grasping)
    if len(masks_set) > 0:
        print("No valid bimanual grasp found, returning single arm grasp")
        return [int(masks_set.pop())]
    return -1

def check_valid_bimanual_grasp(pair_indices, out, viz_mask_pair = True):
    mask_pair = out['masks'][list(pair_indices)]
    img_width = mask_pair.shape[3]
    
    if viz_mask_pair:
        plt.imshow(mask_pair[0].squeeze().cpu().numpy())
        plt.imshow(mask_pair[1].squeeze().cpu().numpy(), alpha=0.5)
        plt.show()
        
    obb1, _, _ = get_oriented_bounding_box(mask_pair[0].squeeze().cpu().numpy())
    obb2, _, _ = get_oriented_bounding_box(mask_pair[1].squeeze().cpu().numpy())
    
    arm_config_coverage = 0.55 # assume 63% of the image width is reachable by each arm
    
    obb1_left_graspable = obb1[0][0] < img_width * arm_config_coverage
    obb1_right_graspable = obb1[0][0] > img_width * (1 - arm_config_coverage)
    obb2_left_graspable = obb2[0][0] < img_width * arm_config_coverage
    obb2_right_graspable = obb2[0][0] > img_width * (1 - arm_config_coverage)
    
    distance = np.linalg.norm(np.array(obb1[0]) - np.array(obb2[0]))
    print(f"Distance: {distance}")
    
    # Check if the two bounding boxes are far enough apart to avoid collision
    if distance < 600:
        print("Distance too close, moving to next pair")
        return False, False
    
    # Check if the masks are graspable simultaneously
    if (obb1_left_graspable and obb2_right_graspable):
        return True, False
    if (obb1_right_graspable and obb2_left_graspable):
        return True, True # flip left and right grasp mask idx
    
    return False, False
    
def main():
    
    endpt_model = EndpointDataloader(use_hub_detect=True)
    detic_loader = DeticDataloader()
    detic_loader.create()
    detic_loader.default_vocab()
    tracer = TracerDataloader()

    # load img from filepath
    imgpath = '/home/justinyu/multicable-decluttering/decluttering/scripts/dev/outputs/021925_1449/iter_1_raw_image.png'
    img = cv2.imread(imgpath)
    
    plt_path = f"tier_{TIER}/{curr_time}"
    # with open(txt_path, "a") as f:
    #     f.write(f"Iter {i}:\n")

    print("Detecting endpoints...")
    
    bool_detic_mask, detic_mask, out = get_detic_masks(img, detic_loader)

    endpoints = predict_pts(img, endpt_model, bool_detic_mask, detic_mask, plt_path)
    
    tracer.tracer.bool_detic_mask = dilate_masks(bool_detic_mask, 15)
    tracer.tracer.detic_mask = dilate_masks(detic_mask, 15).clone()
    
    
    
    img_down = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))  # (1060, 1904, 3)
    trace_list, endpt_table, densities, mask_num = get_trace_list(img_down, endpoints, tracer, viz=False, bimanual_declutter=BIMANUAL_DECLUTTER)
    print(mask_num)
    
    if BIMANUAL_DECLUTTER:
        mask_num = bimanual_filter_grasps(mask_num, out)
        

    output = get_all_div_points(img_down, endpoints, trace_list, endpt_table, densities, viz=False, name="")
    ip_moves_plot(output, img_down, img, plt_path)
    asc_endpts = output['asc_endpts']

            
if __name__ == "__main__":
    main()
