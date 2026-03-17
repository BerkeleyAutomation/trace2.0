import cv2
import time
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
# from untangling.utils.tcps import *

# [Claude bugfix 2/16/26] YK was previously provided by `from untangling.utils.tcps import *`
# which is no longer available after the refactor. RigidTransform just needs the frame name strings,
# which match the TCP frame names used in yumi_jacobi.interface.AsyncInterface.__init__.
class _YKFrames:
    r_tcp_frame = "r_tcp_frame"
    l_tcp_frame = "l_tcp_frame"
YK = _YKFrames()
from skimage.filters import frangi
from skimage.morphology import skeletonize
from divergencecopy import *
from autolab_core.transformations import rotation_matrix, quaternion_from_matrix
# from untangling.utils.interface_rws_BRIO import CameraIntrinsics
from autolab_core import RigidTransform, Point
from autolab_core import CameraIntrinsics

X_BUFFER = 250
Y_BUFFER = 100


def out_bounds(pt, img):
    # is_out = pt[0] < Y_BUFFER or pt[0] > img.shape[0] - Y_BUFFER or pt[1] < X_BUFFER or pt[1] > img.shape[1] - X_BUFFER
    is_out = pt[0] < Y_BUFFER or pt[0] > 600 - Y_BUFFER or pt[1] < X_BUFFER or pt[1] > img.shape[1] - X_BUFFER
    src = []
    if is_out:
        if pt[0] < Y_BUFFER:
            src.append('top')
        if pt[0] > img.shape[0] - Y_BUFFER:
            src.append('bottom')
        if pt[1] < X_BUFFER:
            src.append('left')
        if pt[1] > img.shape[1] - X_BUFFER:
            src.append('right')
    return is_out, src


DIST_TO_TABLE_LEFT  = 1.025 # meters
DIST_TO_TABLE_RIGHT = 0.99 # meters

T_CAM_BASE = RigidTransform.load("/home/justinyu/multicable-decluttering/decluttering/scripts/brio/brio_to_world_bww.tf").as_frames(from_frame="brio", to_frame="base_link")

CAM_INTR = CameraIntrinsics(fx=3.43246678e+03, fy=3.44478930e+03,
           cx=1.79637288e+03, cy=1.08661527e+03, width=3840, height=2160, frame='brio')

FIXED_DEPTH = 0.061
FOAM_DEPTH = 0.0582

# Hardcoded [x, y] offset (meters, world frame) applied to all robot target positions.
# Adjust these values to shift the robot's target x,y positions globally.
XY_OFFSET = [-0.03, 0.02]

FOAM_DEPTH_R_ADJ_VAL =  -0.0025
FOAM_DEPTH_L_ADJ_VAL = -0.005

FOAM_DEPTH_R = 0.0543 + FOAM_DEPTH_R_ADJ_VAL
FOAM_DEPTH_L = 0.0534 + FOAM_DEPTH_L_ADJ_VAL

DEC_FIXED_DEPTH_OBJ_L = 0.0605 #0.0650
DEC_FIXED_DEPTH_OBJ_R = 0.0605 #0.0650
DEC_CAM_INTR = CameraIntrinsics(fx=3.43246678e+03, fy=3.44478930e+03,
           cx=1.79637288e+03, cy=1.08661527e+03, width=3840, height=2160, frame='brio')

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.04-FOAM_DEPTH])
    point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
    # print(point_3d_world)
    point_3d_world = point_3d_world[:3]/point_3d_world[3]
    point_3d_world[0] = 0.95-point_3d_world[0]
    # point_3d_world[0] += XY_OFFSET[0]
    point_3d_world[1] += XY_OFFSET[1]
    # print("HELLO HELLO")
    print(point_3d_world)
    point_3d_world[-1] = FOAM_DEPTH
    # print('non-homogenous = ', point_3d_world)
    return point_3d_world

def visualize_trace(img, trace):
    img = img.copy()
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
    for i in range(len(trace) - 1):
        # if trace is ordered dict, use below logic
        if not isinstance(trace, OrderedDict):
            pt1 = tuple(trace[i].astype(int))
            pt2 = tuple(trace[i+1].astype(int))
        else:
            trace_keys = list(trace.keys())
            pt1 = trace_keys[i]
            pt2 = trace_keys[i + 1]
        cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(trace)), 4)
    plt.title("Trace Visualized")
    plt.imshow(img)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

def gaussian_2d(width, height):
    x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.8, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def is_valid_cc(cc_stats, cc_labels, cc, radius=25):
    if cc == 0:
        return False

    return is_large_cc(cc_stats, cc)

def is_large_cc(cc_stats, cc):
    return cc_stats[cc][4] > 400 # minimum cc pixel area of 1000 may be tuned

# checks that the CC doesn't go from one end of the image to the other
def not_touching_both_ends_cc(cc_lables, cc):
    if cc in cc_lables[0, :] and cc in cc_lables[-1, :]:
        return False
    if cc in cc_lables[:, 0] and cc in cc_lables[:, -1]:
        return False
    return True

# Point of inaccessability is the point within a polygon that is furthest from the boundary of the polygon
def get_pole(labels, label):
    poles = None

    #for label in range(num_labels):
    padded_labels = np.pad(labels, ((1,1),(1,1)), mode='constant', constant_values=0)

    binary_img = (padded_labels == label).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)

    #plot gaussian
    gaussian = gaussian_2d(padded_labels.shape[0], padded_labels.shape[1])
    # plt.clf()
    # plt.imshow(gaussian)
    # plt.show()
    center_bias = np.multiply(gaussian_2d(padded_labels.shape[0], padded_labels.shape[1]), dist_transform)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(center_bias)
    adjusted_max_loc = (max_loc[0]-1, max_loc[1]-1)
    poles = adjusted_max_loc

    return poles

def get_dist_gradient(labels, label):
    poles = None

    #for label in range(num_labels):
    padded_labels = np.pad(labels, ((1,1),(1,1)), mode='constant', constant_values=0)

    binary_img = (padded_labels == label).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)

    import matplotlib.pyplot as plt
    import pdb; pdb.set_trace()
    
    #plot gaussian
    gaussian = gaussian_2d(padded_labels.shape[0], padded_labels.shape[1])
    # plt.clf()
    # plt.imshow(gaussian)
    # plt.show()
    # center_bias = np.multiply(gaussian_2d(padded_labels.shape[0], padded_labels.shape[1]), dist_transform)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(center_bias)
    # adjusted_max_loc = (max_loc[0]-1, max_loc[1]-1)
    # poles = adjusted_max_loc

    return poles

def get_tangent_vec_poi(push_coord, trace):
    
    poi_trace = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]
    poi_closest = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]
    return poi_trace, poi_closest

    
def get_poi_and_vec_for_push(push_coord, img_rgb, trace, place_box_img_W=150, edge_case=False, viz=False, y_buffer=0, x_buffer=0):
    push_coord = [push_coord[1], push_coord[0]]
    poi_trace = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)]
    poi_trace = [poi_trace[1], poi_trace[0]]
    poi_closest = trace[get_closest_trace_idx([push_coord[1],push_coord[0]], trace)+1]
    poi_closest = [poi_closest[1], poi_closest[0]]

    # if viz:
    #     plt.scatter(poi_trace[0], poi_trace[1])
    #     plt.scatter(poi_closest[0], poi_closest[1])
    #     plt.imshow(img_rgb)
    #     mng = plt.get_current_fig_manager()
    #     mng.full_screen_toggle()
    #     plt.show()
    
    poi_tan_vec = (np.array(poi_trace) - np.array(poi_closest))/np.linalg.norm(np.array(poi_trace) - np.array(poi_closest)) #unit vector

    x_start = max(0,poi_trace[1]-place_box_img_W//2)
    x_end = min(poi_trace[1]+place_box_img_W//2, img_rgb.shape[0])
    y_start = max(0, poi_trace[0]-place_box_img_W//2)
    y_end = min(img_rgb.shape[1], poi_trace[0]+place_box_img_W//2)

    cropped_img = img_rgb[x_start:x_end, y_start:y_end, 0]
    poi_cropped = [poi_trace[0]-y_start, poi_trace[1]-x_start]

    thresh, maxValue = 127, 255
    _, binary_img = cv2.threshold(cropped_img, thresh, maxValue, cv2.THRESH_BINARY)


    dilate_size = 4
    dilate_kernel = np.ones((dilate_size,dilate_size), np.uint8)
    dilated_binary_img = cv2.dilate(binary_img, dilate_kernel)

    for i in range(dilated_binary_img.shape[0]):
        for j in range(dilated_binary_img.shape[1]):
            if dilated_binary_img[i][j] == 255:
                dilated_binary_img[i][j] = 0
            else:
                dilated_binary_img[i][j] = 1

    num_labels, labels, stats, centroids= cv2.connectedComponentsWithStats(dilated_binary_img)
    plt.imshow(labels)
    poidotproducts = []
    candidate_pts = []
    for label in range(num_labels):
        if is_valid_cc(stats, labels, label): # minimum cc pixel area of 1000 may be tuned
            center_point = get_pole(labels=labels, label = label)

            pointerest_center_vec = (np.array(center_point) - poi_cropped)/np.linalg.norm(np.array(center_point) - poi_cropped)
                        
            poidotproducts.append(np.dot(poi_tan_vec, pointerest_center_vec))
            candidate_pts.append(center_point)
            plt.scatter(center_point[0], center_point[1], s=7)

    if viz:
        plt.show()
    else:
        plt.clf()
    def check_in_buffer(pt):
        pt = [pt[1], pt[0]]

        if pt[0] > (img_rgb.shape[0] - y_buffer):
            return True
        if pt[0] < y_buffer:
            return True
        if pt[1] > (img_rgb.shape[1] - x_buffer):
            return True
        if pt[1] < x_buffer:
            return True
        return False

    if edge_case is True:
        keep_candidate_pts = []
        keep_poidotproducts = []
        for pt, poidotproduct in zip(candidate_pts, poidotproducts):
            if check_in_buffer(pt):
                keep_candidate_pts.append(pt)
                keep_poidotproducts.append(poidotproduct)
        candidate_pts = keep_candidate_pts
        poidotproducts = keep_poidotproducts
        chosen_pt = candidate_pts[poidotproducts.index(min(poidotproducts))]
    else:
        chosen_pt = candidate_pts[poidotproducts.index(max(poidotproducts))]

    cen_poi_vec = (np.array(chosen_pt) - poi_cropped)
    
    return poi_trace, cen_poi_vec


def get_closest_trace_idx(poi, trace):
    return np.argmin(np.linalg.norm(np.array(trace) - np.array(poi)[None, ...], axis=1))

## H
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
        
        _, labels, _,_ = cv2.connectedComponentsWithStats(dilated_binary_img)
        dist_transform = cv2.distanceTransform(dilated_binary_img, cv2.DIST_L2, 5)
        
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


def get_pindown_gripper_rot(poi, pt1, pt2):
    yaw = np.arctan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    gripper_rot = rotation_matrix(-yaw, [0, 0, 1], poi)[:3,:3]
    return gripper_rot

def color_for_pct(pct):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(pct, 1, 1))

def visualize_trace(img, trace, save=False):
    img_copy = img.copy()
    for i in range(len(trace) - 1):
        pt1, pt2 = get_trace_points(trace, i)
        cv2.line(img_copy, pt1[::-1], pt2[::-1], color_for_pct(i / len(trace)), 4)
    display_image(img_copy, trace, save)

def get_trace_points(trace, idx):
    if not isinstance(trace, OrderedDict):
        return tuple(trace[idx].astype(int)), tuple(trace[idx+1].astype(int))
    trace_keys = list(trace.keys())
    return trace_keys[idx], trace_keys[idx + 1]

def display_image(img, trace, save):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Trace Visualized")
    if save:
        plt.savefig("multicable_baseline.png")
    else:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

def perform_pindown(trace, endpt_coord, img, iface, pin_arm=None, push_coord = None):
    iface.close_grippers()

    if pin_arm == 'right':

        iface.open_gripper('right')
        iface.sync()
        #idx_keep is all indeces of the trace the right side of the image
        radoe = [np.linalg.norm(trace[i]- push_coord) for i in range(len(trace))]
        idx_keep = [i for i in range(len(trace)) if trace[i][1] > 590 and radoe[i]>180]
        #dists is a list of all the distances from the endpt_coord to each point in the trace
        dists = [np.linalg.norm(trace[i]-endpt_coord) for i in idx_keep]
        #if there are no potential points, return None
        if len(idx_keep)==0:
            return None
        trace_idx, pindown_pt = None, None
        for min_idx in np.argsort(dists):
            trace_idx = idx_keep[min_idx]
            pindown_pt = trace[trace_idx]
            
            #if pindown_pt[0] < 1032-X_BUFFER or pindown_pt[0] > img.shape[0] - X_BUFFER or pindown_pt[1] < y_buffer or pindown_pt[1] > img.shape[1] - y_buffer:
            #checking for a usable pindown point by checking that it doesn't fall within out_bounds which covers the area where the robot can operate
            if out_bounds(pindown_pt, img)[0]:
                    trace_idx, pindown_pt = None, None
            else:
                    break
            

        # assert pindown_pt is not None and trace_idx is not None
        if pindown_pt is None and trace_idx is None:
            return None
        pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]
        print('pindown_pt: ', pindown_pt)
        #i think all the below code is repetitive so i'm going to leave it commented out -abby
        #radeo is a list of all the distances from the endpt_coord to each point in the trace
        #radoe = [np.linalg.norm(trace[i]-endpt_coord) for i in range(len(trace))]
        #for min_idx in np.argsort(dists):
            #trace_idx = idx_keep[min_idx]
            #pindown_pt = trace[trace_idx] 
            #idx_keep is list of all the points that satisfy the bound  
            #trace_idx = idx_keep[min_idx]
            # pindown_pt = trace[trace_idx]
            # pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]
            # if not check_in_buffer(pindown_pt, y_buffer, x_buffer, img.color._data.shape):
            #     continue
            # break
        # trace_idx = idx_keep[keep_normalized_cable_density.index(min(keep_normalized_cable_density))]
        # pindown_pt = trace[trace_idx]
        # pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]

        # # visualize_trace(img.color._data, [trace[i] for i in idx_keep])
        # # plt.imshow(img.color._data)
        # # for i in [trace[i] for i in idx_keep]:
        # #     plt.scatter(i[1],i[0], c='b', s=1)
        # # plt.scatter(pindown_pt[0], pindown_pt[1], c='r')
        # # plt.title('pindown pt')
        # # plt.show()
 
        place1 = get_world_coord_from_pixel_coord(pindown_pt, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FIXED_DEPTH]
        try:
            x1, x2 = get_world_coord_from_pixel_coord(trace[trace_idx-1], CAM_INTR), get_world_coord_from_pixel_coord(trace[trace_idx+1], CAM_INTR)
        except:
            trace_idx_min = max(0, trace_idx-1)
            trace_idx_max = min(len(trace)-1, trace_idx+1)
            x1, x2 = get_world_coord_from_pixel_coord(trace[trace_idx_min], CAM_INTR), get_world_coord_from_pixel_coord(trace[trace_idx_max], CAM_INTR)
        gripper_rot = get_pindown_gripper_rot(place1, x1, x2)

        intermediate_place1 = place1 + np.array([0, 0, 0.07])

        gripper_rot = gripper_rot @ iface.GRIP_DOWN_R

        # left arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        good_path = iface.check_cartesian_path(r_targets=[intermediate_place1_transform, place1_transform], removejumps=[6])
        if not good_path:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")

        iface.go_cartesian(r_targets=[intermediate_place1_transform,place1_transform], removejumps=[6])
        iface.sync()
        time.sleep(2)
        # iface.open_gripper('left')
        # iface.go_cartesian(l_targets=[intermediate_place1_transform], removejumps=[6])
        # iface.close_gripper('left')
        # iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
        # iface.open_gripper('left')
        # iface.sync()

        # iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
        # iface.sync()

        iface.close_gripper('right')
        iface.sync()

    else:
        iface.open_gripper('left')
        iface.sync()
        # radoe = [np.linalg.norm(trace[i]-endpt_coord) for i in range(len(trace))]
        # idx_keep = [i for i in range(len(trace)) if (trace[i][1] > 590 and radoe[i]<220 and radoe[i]>150)]
        # # idx_keep = [i for i in range(len(trace)) if trace[i][1] > 590]
        # if len(idx_keep)==0:
        #     return None
        # keep_normalized_cable_density = [normalized_cable_density[i] for i in idx_keep]
        # trace_idx, pindown_pt = None, None
        # for min_idx in np.argsort(keep_normalized_cable_density):
        #     trace_idx = idx_keep[min_idx]
        #     pindown_pt = trace[trace_idx]
        #     # if pindown_pt[0] < x_buffer or pindown_pt[0] > img.shape[0] - x_buffer or pindown_pt[1] < y_buffer or pindown_pt[1] > img.shape[1] - y_buffer:
        #     if out_bounds(pindown_pt, img)[0]:
        #         trace_idx, pindown_pt = None, None
        #         continue
        #     break
        radoe = [np.linalg.norm(trace[i]- push_coord) for i in range(len(trace))]
        idx_keep = [i for i in range(len(trace)) if trace[i][1] <= 590 and radoe[i]>180]
        dists = [np.linalg.norm(trace[i]-endpt_coord) for i in idx_keep]
        if len(idx_keep)==0:
            return None
        trace_idx, pindown_pt = None, None
        for min_idx in np.argsort(dists):
            trace_idx = idx_keep[min_idx]
            pindown_pt = trace[trace_idx]
            # if pindown_pt[0] < x_buffer or pindown_pt[0] > img.shape[0] - x_buffer or pindown_pt[1] < y_buffer or pindown_pt[1] > img.shape[1] - y_buffer:
            if out_bounds(pindown_pt, img)[0]:
                trace_idx, pindown_pt = None, None
                continue
            break

        assert pindown_pt is not None and trace_idx is not None
        pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]
        print('pindown_pt: ', pindown_pt)
        # trace_idx = idx_keep[keep_normalized_cable_density.index(min(keep_normalized_cable_density))]
        
        # pindown_pt = trace[trace_idx]
        # pindown_pt = [int(pindown_pt[1]), int(pindown_pt[0])]

        # plt.imshow(img.color._data)
        # plt.scatter(pindown_pt[0], pindown_pt[1])
        # plt.title('pindown pt')
        # plt.show()

        place1 = get_world_coord_from_pixel_coord(pindown_pt, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FIXED_DEPTH]
        intermediate_place1 = place1 + np.array([0, 0, 0.07])

        x1, x2 = get_world_coord_from_pixel_coord(trace[trace_idx-1], CAM_INTR), get_world_coord_from_pixel_coord(trace[trace_idx+1], CAM_INTR)
        gripper_rot = get_pindown_gripper_rot(place1, x1, x2)

        intermediate_place1 = place1 + np.array([0, 0, 0.07])

        gripper_rot = gripper_rot @ iface.GRIP_DOWN_R

        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        place1_transform = RigidTransform(
            translation=place1,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        good_path = iface.check_cartesian_path(l_targets=[intermediate_place1_transform, place1_transform], removejumps=[6])
        if not good_path:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")
        iface.go_cartesian(l_targets=[intermediate_place1_transform,place1_transform], removejumps=[6])
        # iface.open_gripper('right')
        # iface.go_cartesian(r_targets=[intermediate_place1_transform], removejumps=[6])
        # iface.close_gripper('right')
        # iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
        # iface.open_gripper('right')
        # iface.sync()
        # iface.sync()
        iface.sync()
        time.sleep(2)

        # iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
        # iface.sync()

        iface.close_gripper('left')
        iface.sync()
    
    iface.sync()
    time.sleep(2)

    return pin_arm

def perform_push_through(poi_trace, cen_poi_vec, iface, viz=False, pin_arm=None):
    iface.home()
    iface.close_grippers()

    waypoint1 = cen_poi_vec
    waypoint2 = poi_trace

    # print('waypoint1: ', waypoint1)
    # print('waypoint2: ', waypoint2)
    print("PERFORMING PUSH THROUGH")
    
    place1 = get_world_coord_from_pixel_coord(waypoint1, CAM_INTR) # convert pixel coordinates to 3d coordinates
    place2 = get_world_coord_from_pixel_coord(waypoint2, CAM_INTR) # convert pixel coordinates to 3d coordinates
    if poi_trace[0] > 1904:
        place1[1] += 0.02
        place2[1] += 0.02
        place1 = np.array([place1[0], place1[1], FOAM_DEPTH_R])
        place2 = np.array([place2[0], place2[1], FOAM_DEPTH_R])
        print(place1)
        print(place2)
        intermediate_place1 = place1 + np.array([0, 0, 0.07])
        intermediate_place2 = place2 + np.array([0, 0, 0.07])
        print(intermediate_place1)
        print(intermediate_place2)

        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        current_tf = iface.get_FK('right')
        
        # add an interpolated RigidTransform between current_tf and intermediate_place1_transform
        curr_intermediate_transform = RigidTransform(
            translation = current_tf.translation + (intermediate_place1_transform.translation - current_tf.translation) / 2,
            rotation = current_tf.rotation,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        
        place1_transform = RigidTransform(
            translation=place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        
        # m1 = iface.listRT2Motion(robot = iface.yumi.right, start = iface.driver_right.current_joint_position, wp_list = [curr_intermediate_transform, intermediate_place1_transform, place1_transform])
        # lm1 = iface.listRT2LinearMotion(robot = iface.yumi.right, start = place1_transform, goal = place2_transform)
        # motion = [m1, lm1]
        
        traj = iface.plan_linear_waypoints(r_targets=[intermediate_place1_transform, place1_transform, place2_transform], return_motions=False)
        
        # traj = iface.plan(motion)
        # import pdb; pdb.set_trace()
        if traj is None:
            print("Motion planning failed!")
            return RuntimeError
        
        result = iface.run_trajectories(traj) # come back to
            
        iface.go_delta(right_delta=[0, 0, 0.05])

    else:
        place1 = np.array([place1[0], place1[1], FOAM_DEPTH_L])
        place2 = np.array([place2[0], place2[1], FOAM_DEPTH_L])
        print(place1)
        print(place2)
        intermediate_place1 = place1 + np.array([0, 0, 0.07])
        intermediate_place2 = place2 + np.array([0, 0, 0.07])
        print(intermediate_place1)
        print(intermediate_place2)

        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        
        current_tf = iface.get_FK('left')
        
        # add an interpolated RigidTransform between current_tf and intermediate_place1_transform
        curr_intermediate_transform = RigidTransform(
            translation = current_tf.translation + (intermediate_place1_transform.translation - current_tf.translation) / 2,
            rotation = current_tf.rotation,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        
        place1_transform = RigidTransform(
            translation=place1,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        
        # m1 = iface.listRT2Motion(robot = iface.yumi.left, start = iface.driver_left.current_joint_position, wp_list = [curr_intermediate_transform, intermediate_place1_transform, place1_transform])
        # lm1 = iface.listRT2LinearMotion(robot = iface.yumi.left, start = place1_transform, goal = place2_transform)
        # motion = [m1, lm1]
        
        traj = iface.plan_linear_waypoints(l_targets=[intermediate_place1_transform, place1_transform, place2_transform], return_motions=False)

        # traj = iface.plan(motion)
        if traj is None:
            print("Motion planning failed!")
            return RuntimeError

        result = iface.run_trajectories(traj)
        
        iface.go_delta(left_delta=[0, 0, 0.05])

    # if pin_arm is not None:
    #     time.sleep(1)
    #     iface.open_gripper(pin_arm)
    #     iface.sync()
    
    iface.home()
    iface.close_grippers()
    return True


def perform_pick_away(poi_trace, cen_poi_vec, iface, trace, deviaiton_idx, viz=False):
    
    # waypoint1 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec
    # waypoint2 = np.array([poi_trace[0], poi_trace[1]]) - cen_poi_vec*0.7
    waypoint1 = np.array([poi_trace[0], poi_trace[1]])
    waypoint2 = np.array([poi_trace[0], poi_trace[1]]) + cen_poi_vec * 2

    # iface = fullPipeline.iface

    # print(h)
    waypoint1 = [int(waypoint1[1]), int(waypoint1[0])]
    waypoint2 = [int(waypoint2[1]), int(waypoint2[0])]

    print('waypoint1: ', waypoint1)
    print('waypoint2: ', waypoint2)
    

    # fixed_depth = 0.0438
    if poi_trace[0] > 1904:
        place1 = get_world_coord_from_pixel_coord(waypoint1, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FIXED_DEPTH]
        place2 = [place2[0], place2[1], FIXED_DEPTH]
        # print(place1)
        # print(place2)
        x1, x2 = get_world_coord_from_pixel_coord(trace[deviaiton_idx], CAM_INTR), get_world_coord_from_pixel_coord(trace[deviaiton_idx-1], CAM_INTR)
        gripper_rot = get_pindown_gripper_rot(place1, x1, x2)
        gripper_rot = gripper_rot @ iface.GRIP_DOWN_R
        intermediate_place1 = place1 + np.array([0, 0, 0.07])
        intermediate_place2 = place2 + np.array([0, 0, 0.07])
        intermediate_place3 = np.array([.4, -.25, 0.12])
        # right arm
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

        intermediate_place2_transform = RigidTransform(
            translation=intermediate_place2,
            # rotation= iface.GRIP_DOWN_R,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        intermediate_place3_transform = RigidTransform(
            translation=intermediate_place3,
            # rotation= iface.GRIP_DOWN_R,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        place1_transform = RigidTransform(
            translation=place1,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        # place1_transform_up = RigidTransform(
        #     translation=place1 + np.array([0, 0, 0.0]),
        #     rotation= gripper_rot,
        #     from_frame=YK.r_tcp_frame,
        #     to_frame="base_link",
        # )
        # place2_transform_up = RigidTransform(
        #     translation=place2 + np.array([0, 0, 0.02]),
        #     rotation= gripper_rot,
        #     from_frame=YK.r_tcp_frame,
        #     to_frame="base_link",
        # )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= gripper_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        # print(intermediate_place1_transform, place1_transform)
        good_path = iface.check_cartesian_path(r_targets=[intermediate_place1_transform, place1_transform, place2_transform, 
                                                           intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        if not good_path:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")
        
        iface.open_gripper('right')
        iface.sync()
        time.sleep(2)
        iface.go_cartesian(r_targets=[intermediate_place1_transform, place1_transform], removejumps=[6])
        iface.close_gripper('right')
        iface.sync()
        time.sleep(2)
        iface.go_cartesian(r_targets=[place2_transform], removejumps=[6])
        iface.open_gripper('right')
        iface.sync()
        time.sleep(2)
        iface.go_cartesian(r_targets = [intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        iface.sync()
        time.sleep(2)

    else:
        place1 = get_world_coord_from_pixel_coord(waypoint1, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place2 = get_world_coord_from_pixel_coord(waypoint2, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FIXED_DEPTH]
        place2 = [place2[0], place2[1], FIXED_DEPTH]
        # print(place1)
        # print(place2)
        x1, x2 = get_world_coord_from_pixel_coord(trace[deviaiton_idx], CAM_INTR), get_world_coord_from_pixel_coord(trace[deviaiton_idx-1], CAM_INTR)
        gripper_rot = get_pindown_gripper_rot(place1, x1, x2)
        gripper_rot = gripper_rot @ iface.GRIP_DOWN_R
        intermediate_place1 = place1 + np.array([0, 0, 0.07])
        intermediate_place2 = place2 + np.array([0, 0, 0.07])
        intermediate_place3 = np.array([0.4, 0.25, 0.12])
        
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        intermediate_place2_transform = RigidTransform(
            translation=intermediate_place2,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        intermediate_place3_transform = RigidTransform(
            translation=intermediate_place3,
            rotation= iface.GRIP_DOWN_R,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        place1_transform = RigidTransform(
            translation=place1,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        # place1_transform_up = RigidTransform(
        #     translation=place1 + np.array([0, 0, 0.04]),
        #     rotation= gripper_rot,
        #     from_frame=YK.l_tcp_frame,
        #     to_frame="base_link",
        # )
        # place2_transform_up = RigidTransform(
        #     translation=place2 + np.array([0, 0, 0.04]),
        #     rotation= gripper_rot,
        #     from_frame=YK.l_tcp_frame,
        #     to_frame="base_link",
        # )
        place2_transform = RigidTransform(
            translation=place2,
            rotation= gripper_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        # good_path = iface.check_cartesian_path(l_targets=[intermediate_place1_transform, place1_transform, place1_transform_up, place2_transform_up, place2_transform, 
        #                                                    intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        good_path = iface.check_cartesian_path(l_targets=[intermediate_place1_transform, place1_transform, place2_transform, 
                                                           intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        if not good_path:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")

        iface.open_gripper('left')
        iface.sync()
        time.sleep(2)
        iface.go_cartesian(l_targets=[intermediate_place1_transform, place1_transform], removejumps=[6])
        iface.close_gripper('left')
        iface.sync()
        time.sleep(2)
        iface.go_cartesian(l_targets=[place2_transform], removejumps=[6])
        iface.sync()
        time.sleep(2)
        iface.open_gripper('left')
        iface.sync()
        time.sleep(2)
        iface.go_cartesian(l_targets = [intermediate_place2_transform, intermediate_place3_transform], removejumps=[6])
        iface.sync()
        time.sleep(2)
    
    iface.open_grippers()
    iface.sync()
    time.sleep(2)
    iface.home()
    iface.sync()
    time.sleep(2)
    iface.close_grippers()
    iface.sync()
    time.sleep(2)

def get_gripper_rot(yaw, iface):
    gripper_rot = rotation_matrix(yaw, [0, 0, 1])
    gripper_rot = gripper_rot[:3,:3] @ iface.GRIP_DOWN_R
    # print(gripper_rot)
    return gripper_rot

def drop_in_bin(iface, arm = 'left'):
    # iface = fullPipeline.iface
    if arm == 'left':
        place1 = [0.5327735, 0.56311663, 0.26703631]
        rot = np.array([[ 0.98123146,  0.1333951,   0.13925002], [ 0.06947921, -0.91819131,  0.38999661], [ 0.1798818,  -0.37300196, -0.91022639]])
        place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian(l_targets=[place1_transform], removejumps=[6])
        iface.sync()
    elif arm == 'right':
        place1 = [0.48272088, -0.55385822, 0.26748793]
        rot = np.array([[ 0.96381043, -0.19505362, -0.18172382], [-0.21366951, -0.97283961, -0.08904179], [-0.15942021,  0.12464825, -0.97930997]])
        place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian(r_targets=[place1_transform], removejumps=[6])
        iface.sync()
    elif arm == 'both':
        l_place1 = [0.5327735, 0.56311663, 0.26703631]
        l_rot = np.array([[ 0.98123146,  0.1333951,   0.13925002], [ 0.06947921, -0.91819131,  0.38999661], [ 0.1798818,  -0.37300196, -0.91022639]])
        r_place1 = [0.48272088, -0.55385822, 0.26748793]
        r_rot = np.array([[ 0.96381043, -0.19505362, -0.18172382], [-0.21366951, -0.97283961, -0.08904179], [-0.15942021,  0.12464825, -0.97930997]])
        l_place1_transform = RigidTransform(
            translation=l_place1,
            rotation=l_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        r_place1_transform = RigidTransform(
            translation=r_place1,
            rotation=r_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian(r_targets=[r_place1_transform], l_targets=[l_place1_transform], removejumps=[6])
        iface.sync()


    time.sleep(1)
    iface.open_grippers()

def pipeline_drop_in_bin(iface, arm = 'left'):
    # iface = fullPipeline.iface
    if arm == 'left':
        place1 = [0.5327735, 0.56311663, 0.26703631]
        rot = np.array([[ 0.98123146,  0.1333951,   0.13925002], [ 0.06947921, -0.91819131,  0.38999661], [ 0.1798818,  -0.37300196, -0.91022639]])
        place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian_waypoints(l_targets=[place1_transform])
    elif arm == 'right':
        place1 = [0.48272088, -0.55385822, 0.26748793] # [0.48272088, -0.55385822, 0.26748793]
        rot = np.array([[ 0.96381043, -0.19505362, -0.18172382], [-0.21366951, -0.97283961, -0.08904179], [-0.15942021,  0.12464825, -0.97930997]])
        place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian_waypoints(r_targets=[place1_transform])
    elif arm == 'both':
        l_place1 = [0.5327735, 0.56311663, 0.26703631]
        l_rot = np.array([[ 0.98123146,  0.1333951,   0.13925002], [ 0.06947921, -0.91819131,  0.38999661], [ 0.1798818,  -0.37300196, -0.91022639]])
        r_place1 = [0.48272088, -0.55385822, 0.26748793]
        r_rot = np.array([[ 0.96381043, -0.19505362, -0.18172382], [-0.21366951, -0.97283961, -0.08904179], [-0.15942021,  0.12464825, -0.97930997]])
        l_place1_transform = RigidTransform(
            translation=l_place1,
            rotation=l_rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        r_place1_transform = RigidTransform(
            translation=r_place1,
            rotation=r_rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        iface.go_cartesian_waypoints(r_targets=[r_place1_transform], l_targets=[l_place1_transform])


    # time.sleep(1)
    iface.open_grippers()

def perform_point_grasp_and_bin_drop_pipeline(center, angle, interface, img = None, viz=False, pin_arm=None):
    
    waypoint1 = np.array([center[0], center[1]])
    iface = interface

    print(waypoint1)
    
    iface.open_grippers()
    if waypoint1[0] > 1750: #prev = 550
        print("Using right arm")
        place1 = get_world_coord_from_pixel_coord(waypoint1, DEC_CAM_INTR) # convert pixel coordinates to 3d coordinates
        # place2 = get_world_coord_from_pixel_coord(waypoint2, iface.cam.intrinsics) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], DEC_FIXED_DEPTH_OBJ_R]
        intermediate_place1 = [place1[0], place1[1], 0.26]

        print(f"{angle=}")
        rot = get_gripper_rot(np.radians(angle), iface)
        rm = np.eye(4)
        rm[:3,:3] = rot
        
        print(f"{iface.get_FK('right')=}")
        
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        print(f"{intermediate_place1_transform=}")
        iface.go_cartesian_waypoints(r_targets=[intermediate_place1_transform])
        iface.go_linear_single(r_target=place1_transform)
        
        
        # time.sleep(0.5)
        iface.close_grippers('right')
        # time.sleep(1.0)
        iface.go_linear_single(r_target=intermediate_place1_transform)
        # time.sleep(0.1)
        pipeline_drop_in_bin(iface, arm = 'right') # sleeps commented

    else:
        print("Using left arm")
        place1 = get_world_coord_from_pixel_coord(waypoint1, DEC_CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], DEC_FIXED_DEPTH_OBJ_L]
        intermediate_place1 = [place1[0], place1[1], 0.26]  #prev = 0.15
        # print(f"{place1=}")
        # print(f"{intermediate_place1=}")
        
        print(f"{angle=}")
        rot = get_gripper_rot(np.radians(angle), iface)
        
        rm = np.eye(4)
        rm[:3,:3] = rot
        
        print(f"{iface.get_FK('left')=}")
                
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation=rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        place1_transform = RigidTransform(
            translation=place1,
            rotation= rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        print(f"{intermediate_place1_transform=}")
        iface.go_cartesian_waypoints(l_targets=[intermediate_place1_transform])
        iface.go_linear_single(l_target=place1_transform)
        # time.sleep(0.5)
        iface.close_grippers('left')
        # time.sleep(1.0)
        iface.go_linear_single(l_target=intermediate_place1_transform)
        # time.sleep(0.1)
        pipeline_drop_in_bin(iface, arm = 'left')  # sleeps commented



    # time.sleep(0.5)
    iface.home()
    # time.sleep(0.35)


def bimanual_grasp_and_bin_drop_pipeline(center_l, angle_l, center_r, angle_r, interface, img = None, viz=False, pin_arm=None):
    
    waypoint_l = np.array([center_l[0], center_l[1]])
    waypoint_r = np.array([center_r[0], center_r[1]])
    iface = interface
    
    iface.open_grippers()
    place1_l = get_world_coord_from_pixel_coord(waypoint_l, DEC_CAM_INTR) # convert pixel coordinates to 3d coordinates
    place1_l = [place1_l[0], place1_l[1], DEC_FIXED_DEPTH_OBJ_L]
    intermediate_place1_l = [place1_l[0], place1_l[1], 0.19]
    intermediate_place2_l = [place1_l[0], place1_l[1], 0.17]
    rot_l = get_gripper_rot(np.radians(angle_l), iface)    
    intermediate_place1_transform_l = RigidTransform(
        translation=intermediate_place1_l,
        rotation=rot_l,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    
    intermediate_place2_transform_l = RigidTransform(
        translation=intermediate_place2_l,
        rotation=rot_l,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    
    place1_transform_l = RigidTransform(
        translation=place1_l,
        rotation=rot_l,
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )

    place1_r = get_world_coord_from_pixel_coord(waypoint_r, DEC_CAM_INTR) # convert pixel coordinates to 3d coordinates
    place1_r = [place1_r[0], place1_r[1], DEC_FIXED_DEPTH_OBJ_L]
    intermediate_place1_r = [place1_r[0], place1_r[1], 0.19]  #prev = 0.15
    intermediate_place2_r = [place1_r[0], place1_r[1], 0.17]  #prev = 0.15
    rot_r = get_gripper_rot(np.radians(angle_r), iface)
     
    intermediate_place1_transform_r = RigidTransform(
        translation=intermediate_place1_r,
        rotation=rot_r,
        from_frame=YK.r_tcp_frame,
        to_frame="base_link",
    )
    intermediate_place2_transform_r = RigidTransform(
        translation=intermediate_place2_r,
        rotation=rot_r,
        from_frame=YK.r_tcp_frame,
        to_frame="base_link",
    )
    place1_transform_r = RigidTransform(
        translation=place1_r,
        rotation= rot_r,
        from_frame=YK.r_tcp_frame,
        to_frame="base_link",
    )

    # iface.go_cartesian_bimanual(l_targets=[intermediate_place2_transform_l], r_targets=[intermediate_place2_transform_r])
    motion1 = iface.plan_cartesian_waypoints(l_targets=[intermediate_place2_transform_l], r_targets=[intermediate_place2_transform_r])
    motion2 = iface.plan_linear_waypoints(l_targets=[intermediate_place2_transform_l, place1_transform_l], r_targets=[intermediate_place2_transform_r, place1_transform_r], start_from_current_cfg=False)
    
    print("Planning bimanual motion")
    start_time = time.time()
    trajectories = iface.planner.plan([*motion1, *motion2])
    elapsed_time = (time.time() - start_time)
    print(f"Done planning in {elapsed_time}s, running first bimanual motion")
    iface.run_trajectory(*trajectories[:2])
    print("Running next bimanual motion")
    iface.run_trajectory(*trajectories[2:])
    # time.sleep(0.3)
    # iface.go_cartesian_bimanual(l_targets=[place1_transform_l], r_targets=[place1_transform_r], z_retraction = 0.1)
    
    iface.go_linear_single(l_target=place1_transform_l, r_target=place1_transform_r)
    # time.sleep(0.5)
    iface.close_grippers()
    # time.sleep(1.5)
    # iface.go_cartesian_bimanual(l_targets=[intermediate_place1_transform_l], r_targets=[intermediate_place1_transform_r])
    iface.go_linear_single(l_target=intermediate_place1_transform_l, r_target=intermediate_place1_transform_r)
    time.sleep(0.05)
    pipeline_drop_in_bin(iface, arm = 'both')

    # time.sleep(0.5)
    iface.home()
    # time.sleep(0.35)
    
def perform_point_grasp(center, angle, iface, img = None, viz=False, pin_arm=None):
    
    waypoint1 = np.array([center[0], center[1]])
    # iface = fullPipeline.iface
    gripper_down_axis_angle = np.array([0, 0, -1])

    print(waypoint1)
    iface.open_grippers()
    iface.sync()
    if waypoint1[0] > 3300//2:
        print("Using right arm")
        place1 = get_world_coord_from_pixel_coord(waypoint1, CAM_INTR) # convert pixel coordinates to 3d coordinates
        # place2 = get_world_coord_from_pixel_coord(waypoint2, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FOAM_DEPTH_R]
        intermediate_place1 = [place1[0], place1[1], 0.26]
        print(place1)
        rot = get_gripper_rot(angle, iface)
        print(rot)
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        good_path = iface.check_cartesian_path(r_targets=[intermediate_place1_transform, place1_transform, intermediate_place1_transform], removejumps=[6])
        if not good_path:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")

        iface.go_cartesian(r_targets=[intermediate_place1_transform, place1_transform], removejumps=[6])
        iface.sync()
        time.sleep(1.5)
        
        iface.shake_right_R(gripper_down_axis_angle, 3)

        iface.close_gripper('right')
        iface.sync()
        time.sleep(1)

        iface.go_cartesian(r_targets=[intermediate_place1_transform], removejumps=[6])
        iface.sync()
        time.sleep(1)

        drop_in_bin(iface, arm = 'right')

    else:
        print("Using left arm")
        place1 = get_world_coord_from_pixel_coord(waypoint1, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FOAM_DEPTH_L]
        intermediate_place1 = [place1[0], place1[1], 0.26]
        print(place1)
        rot = get_gripper_rot(angle, iface)
        intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation=rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        place1_transform = RigidTransform(
            translation=place1,
            rotation= rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

        good_path = iface.check_cartesian_path(l_targets=[intermediate_place1_transform, place1_transform, intermediate_place1_transform], removejumps=[6])
        if not good_path:
            raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")

        iface.go_cartesian(l_targets=[intermediate_place1_transform, place1_transform], removejumps=[6])
        iface.sync()
        time.sleep(1.5)

        iface.shake_left_R(gripper_down_axis_angle, 3)

        iface.close_gripper('left')
        iface.sync()
        time.sleep(1)

        iface.go_cartesian(l_targets=[intermediate_place1_transform], removejumps=[6])
        iface.sync()
        time.sleep(1)

        drop_in_bin(iface, arm = 'left')


    time.sleep(1)
    iface.arms_clear_home()
    iface.sync()
    time.sleep(1)


def perform_bimanual_point_grasp(centers, angles, iface, img = None, viz=False, pin_arm=None):
    
    # waypoint1 = np.array([center[0], center[1]])

    rh = np.array([centers[0][0], centers[0][1]])
    lh = np.array([centers[1][0], centers[1][1]])
    # iface = fullPipeline.iface

    # print(waypoint1)
    iface.open_grippers()
    iface.sync()
    if rh[0] > 3300//2:
        print("Using right arm")
        place1 = get_world_coord_from_pixel_coord(rh, CAM_INTR) # convert pixel coordinates to 3d coordinates
        # place2 = get_world_coord_from_pixel_coord(waypoint2, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FOAM_DEPTH_R]
        intermediate_place1 = [place1[0], place1[1], 0.26]
        print(place1)
        rot = get_gripper_rot(angles[0], iface)
        r_intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )
        r_place1_transform = RigidTransform(
            translation=place1,
            rotation=rot,
            from_frame=YK.r_tcp_frame,
            to_frame="base_link",
        )

    if lh[0] < 3300//2:
        print("Using left arm")
        place1 = get_world_coord_from_pixel_coord(lh, CAM_INTR) # convert pixel coordinates to 3d coordinates
        place1 = [place1[0], place1[1], FOAM_DEPTH_L]
        intermediate_place1 = [place1[0], place1[1], 0.26]
        print(place1)
        rot = get_gripper_rot(angles[1], iface)
        l_intermediate_place1_transform = RigidTransform(
            translation=intermediate_place1,
            rotation=rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )
        l_place1_transform = RigidTransform(
            translation=place1,
            rotation= rot,
            from_frame=YK.l_tcp_frame,
            to_frame="base_link",
        )

    good_path = iface.check_cartesian_path(r_targets=[r_intermediate_place1_transform, r_place1_transform, r_intermediate_place1_transform], l_targets = [l_intermediate_place1_transform, l_place1_transform, l_intermediate_place1_transform], removejumps=[6])
    if not good_path:
        raise Exception("Unable to plan jump-free cartesian path despite numerous attempts.")

    iface.go_cartesian(r_targets=[r_intermediate_place1_transform, r_place1_transform], l_targets=[l_intermediate_place1_transform, l_place1_transform], removejumps=[6])
    iface.sync()
    time.sleep(1.5)

    # iface.close_grippers()
    # iface.sync()
    # time.sleep(0.5)

    # iface.go_cartesian(r_targets=[r_intermediate_place1_transform], l_targets=[l_intermediate_place1_transform], removejumps=[6])
    # iface.sync()
    # time.sleep(0.5)

    # drop_in_bin(fullPipeline, arm = 'both')
    
    # time.sleep(0.5)
    # iface.home()
    # iface.sync()
    # time.sleep(0.5)
    

# josh - 11/04/24
def goto_gripper_px(pixel: np.ndarray, iface) -> str:
    iface.home()
    iface.close_grippers()
    
    world = get_world_coord_from_pixel_coord(pixel, CAM_INTR)

    if pixel[0] > 1904:
        world[1] += 0.02
        depth_val  = FOAM_DEPTH_R + 0.0025 # 0.0025 is the offset to ensure the gripper is above the foam in case cables are stacked
        rotation   = iface.GRIP_DOWN_R
        from_frame = YK.r_tcp_frame
        arm = "right"
    else:
        depth_val  = FOAM_DEPTH_L + 0.0025
        rotation   = iface.GRIP_DOWN_L
        from_frame = YK.l_tcp_frame
        arm = "left"
    
    final = np.array([world[0], world[1], depth_val]) # goes into foam
    first = world + np.array([0, 0, 0.05])            # goes above POI

    first_transform = RigidTransform(
        translation=first,
        rotation=rotation,
        from_frame=from_frame,
        to_frame="base_link",
    )
    final_transform = RigidTransform(
        translation=final,
        rotation=rotation,
        from_frame=from_frame,
        to_frame="base_link",
    )
    T = [first_transform, final_transform]
    
    if arm == "right":
        traj = iface.plan_linear_waypoints(r_targets=T, return_motions=False)
    else:
        traj = iface.plan_linear_waypoints(l_targets=T, return_motions=False)

    if traj is None:
        raise ValueError("Motion planning failure in goto_gripper_px")
    else:
        iface.run_trajectories(traj)
        return arm


# josh - 11/04/24
def rotate_gripper(arm: str, iface):
    
    positive_90 = iface.get_joint_positions(arm)
    negative_90 = copy.deepcopy(positive_90)
    
    positive_90[-1] = min(positive_90[-1] + np.pi/2, 2*np.pi)   # rotate wrist angles
    negative_90[-1] = max(positive_90[-1] - np.pi/2, -2*np.pi)

    if arm == "right":
        iface.move_to(right_goal=positive_90)
        iface.move_to(right_goal=negative_90)
    else:
        iface.move_to(left_goal=positive_90)
        iface.move_to(left_goal=negative_90)
