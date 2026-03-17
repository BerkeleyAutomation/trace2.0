"""
motion_pyroki.py — YuMi motion primitives using YuMiROSInterface (yumi_realtime).

Implements the same 6 functions as motion_jacobi.py but uses:
  - yumi_realtime.YuMiROSInterface for robot execution (150Hz EGM streaming)
  - jaxlie for rotation math (quaternions instead of RigidTransform)

Exported functions (matching motion_jacobi.py):
  goto_gripper_px, rotate_gripper, perform_push_through,
  perform_point_grasp_and_bin_drop_pipeline,
  bimanual_grasp_and_bin_drop_pipeline, get_push_coords
"""

import sys
sys.path.append('/home/justinyu/multicable-decluttering/')
sys.path.append('/home/justinyu/multicable-decluttering/trace/yumi_realtime/')

import copy
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi
from skimage.morphology import skeletonize

import jax.numpy as jnp
import jaxlie

from autolab_core import RigidTransform, CameraIntrinsics
from autolab_core.transformations import rotation_matrix

from divergencecopy import *  # KEEPOUT_DIST, SAVE_DIR, WHITE, visualize_multiple_paths, etc.

# ---------------------------------------------------------------------------
# Camera intrinsics and depth constants (same values as motion_jacobi.py)
# ---------------------------------------------------------------------------

## Currently does not use the child directory's camera calibration (*IMPORTANT FOR FUTURE CALIBRATION)

T_CAM_BASE = RigidTransform.load(
    "/home/justinyu/multicable-decluttering/decluttering/scripts/brio/brio_to_world_bww.tf"
).as_frames(from_frame="brio", to_frame="base_link")

CAM_INTR = CameraIntrinsics(
    fx=3.43246678e+03, fy=3.44478930e+03,
    cx=1.79637288e+03, cy=1.08661527e+03,
    width=3840, height=2160, frame='brio'
)
DEC_CAM_INTR = CAM_INTR

FOAM_DEPTH           = 0.0582
FIXED_DEPTH          = 0.061
FOAM_DEPTH_R_ADJ_VAL = -0.0025
FOAM_DEPTH_L_ADJ_VAL = -0.005
FOAM_DEPTH_R         = 0.0543 + FOAM_DEPTH_R_ADJ_VAL
FOAM_DEPTH_L         = 0.0534 + FOAM_DEPTH_L_ADJ_VAL
DEC_FIXED_DEPTH_OBJ_L = 0.0605
DEC_FIXED_DEPTH_OBJ_R = 0.0605
XY_OFFSET            = [-0.03, 0.02]

# ---------------------------------------------------------------------------
# Grip-down quaternions [w, x, y, z] for each arm.
# Correspond to gripper pointing straight down toward the work surface.
# Tune these if the actual grip-down orientation differs.
# ---------------------------------------------------------------------------
GRIP_DOWN_WXYZ_R = np.array([0.0, 1.0, 0.0, 0.0])
GRIP_DOWN_WXYZ_L = np.array([0.0, 1.0, 0.0, 0.0])


## Basically PyRoki is open-loop so we need to estimate times that the robot wil take to update its pose
## before sending the next one

T_HOME    = 4.0   # time to reach home pose
T_MOVE    = 3.0   # time to reach an intermediate waypoint
T_CONTACT = 2.5   # time to reach contact / grasp depth
T_GRIPPER = 1.0   # time for gripper to open / close
T_RETRACT = 2.0   # time to retract after grasp

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

X_BUFFER = 250
Y_BUFFER = 100

# BORROWED FROM MOTION_JACOBI
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


def _rot_matrix_to_wxyz(rot3x3):
    """Convert a 3×3 rotation matrix to a wxyz quaternion via jaxlie."""
    return np.array(jaxlie.SO3.from_matrix(jnp.array(rot3x3, dtype=jnp.float32)).wxyz)


def get_gripper_rot_wxyz(yaw_rad):
    """
    Compute the gripper orientation as wxyz quaternion for a given yaw angle.
    Equivalent to motion_jacobi.get_gripper_rot() but returns a quaternion.
    """
    rot_z       = rotation_matrix(yaw_rad, [0, 0, 1])[:3, :3]
    grip_down   = np.array(jaxlie.SO3(wxyz=jnp.array(GRIP_DOWN_WXYZ_R)).as_matrix())
    combined    = rot_z @ grip_down
    return _rot_matrix_to_wxyz(combined)


def get_closest_trace_idx(poi, trace):
    return np.argmin(np.linalg.norm(np.array(trace) - np.array(poi)[None, ...], axis=1))

## Update from motion_jacobi.py: handle non-unit vectors, and clip floating points to 1
def vector_angle(vec1, vec2):
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def vector_distance(vec1, vec2, axis=None):
    return np.linalg.norm(np.array(vec1) - np.array(vec2), axis=axis)

def vector_direction(vec1, vec2):
    return (np.array(vec1) - np.array(vec2)) / vector_distance(vec1, vec2)


# ---------------------------------------------------------------------------
# I added this function, but if it throws an error just remove it and all implementation
# ---------------------------------------------------------------------------


def _viz_target(iface, name: str, position: np.ndarray, wxyz: np.ndarray):
    """Visualize a planned target as a 3D frame in the viser scene."""
    iface.server.scene.add_frame(
        name, wxyz=wxyz, position=position,
        axes_length=0.05, axes_radius=0.005, origin_radius=0.015,
    )


# ---------------------------------------------------------------------------
# Gripper helpers
# ---------------------------------------------------------------------------

def open_grippers(iface):
    iface.call_gripper('left',  False, True)
    iface.call_gripper('right', False, True)
    time.sleep(T_GRIPPER)


def close_grippers(iface):
    iface.call_gripper('left',  True, True)
    iface.call_gripper('right', True, True)
    time.sleep(T_GRIPPER)


def close_gripper(iface, side):
    iface.call_gripper(side, True, True)
    time.sleep(T_GRIPPER)


def open_gripper(iface, side):
    iface.call_gripper(side, False, True)
    time.sleep(T_GRIPPER)


# ---------------------------------------------------------------------------
# Bin-drop helpers (positions mirror pipeline_drop_in_bin in motion_jacobi.py)
# ---------------------------------------------------------------------------

_BIN_L_POS  = np.array([0.5327735,  0.56311663, 0.26703631])
_BIN_L_ROT  = np.array([[ 0.98123146,  0.1333951,   0.13925002],
                          [ 0.06947921, -0.91819131,  0.38999661],
                          [ 0.1798818,  -0.37300196, -0.91022639]])
_BIN_L_WXYZ = _rot_matrix_to_wxyz(_BIN_L_ROT)

_BIN_R_POS  = np.array([0.48272088, -0.55385822, 0.26748793])
_BIN_R_ROT  = np.array([[ 0.96381043, -0.19505362, -0.18172382],
                          [-0.21366951, -0.97283961, -0.08904179],
                          [-0.15942021,  0.12464825, -0.97930997]])
_BIN_R_WXYZ = _rot_matrix_to_wxyz(_BIN_R_ROT)


def pipeline_drop_in_bin(iface, arm='left'):
    """Move to bin and open gripper. Mirrors pipeline_drop_in_bin in motion_jacobi.py."""
    if arm in ('left', 'both'):
        iface.update_target_pose('left',  _BIN_L_POS, _BIN_L_WXYZ, False, True)
    if arm in ('right', 'both'):
        iface.update_target_pose('right', _BIN_R_POS, _BIN_R_WXYZ, False, True)
    time.sleep(T_MOVE)
    open_grippers(iface)


# ---------------------------------------------------------------------------
# Public API — 6 functions matching motion_jacobi.py
# ---------------------------------------------------------------------------

def goto_gripper_px(pixel: np.ndarray, iface) -> str:
    """
    Move the appropriate arm's gripper to a pixel location on the work surface.
    Returns the arm name used ('left' or 'right').
    """
    ### added a pre_home safety measure (you can delete this if it causes problems)
    iface.pre_home()
    time.sleep(T_RETRACT)

    iface.home()
    time.sleep(T_HOME)
    close_grippers(iface)

    world = get_world_coord_from_pixel_coord(pixel, CAM_INTR)

    if pixel[0] > 1904:
        world[1] += 0.02
        depth_val = FOAM_DEPTH_R + 0.0025
        wxyz      = GRIP_DOWN_WXYZ_R
        arm       = 'right'
    else:
        depth_val = FOAM_DEPTH_L + 0.0025
        wxyz      = GRIP_DOWN_WXYZ_L
        arm       = 'left'

    first = np.array([world[0], world[1], depth_val + 0.05])  # above POI
    final = np.array([world[0], world[1], depth_val])          # at foam depth

    _viz_target(iface, "motion/goto/contact", final, wxyz)
    iface.update_target_pose(arm, first, wxyz, True, True)
    time.sleep(T_MOVE)
    iface.update_target_pose(arm, final, wxyz, True, True)
    time.sleep(T_CONTACT)

    return arm


def rotate_gripper(arm: str, iface):
    """
    Rotate the wrist ±90° to help untangle cables.
    Reads the current EE pose from yumi_realtime's cartesian_pose_L/R and
    applies yaw rotations in place.
    """
    tf = iface.cartesian_pose_R if arm == 'right' else iface.cartesian_pose_L

    pos = np.array([tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z])
    current_wxyz = np.array([tf.transform.rotation.w,
                              tf.transform.rotation.x,
                              tf.transform.rotation.y,
                              tf.transform.rotation.z])

    current_so3 = jaxlie.SO3(wxyz=jnp.array(current_wxyz))
    rot_pos90   = jaxlie.SO3.from_z_radians(jnp.float32( np.pi / 2))
    rot_neg90   = jaxlie.SO3.from_z_radians(jnp.float32(-np.pi / 2))
    wxyz_pos90  = np.array((rot_pos90 @ current_so3).wxyz)
    wxyz_neg90  = np.array((rot_neg90 @ current_so3).wxyz)

    _viz_target(iface, "motion/rotate/pos", pos, wxyz_pos90)
    iface.update_target_pose(arm, pos, wxyz_pos90, True, True)
    time.sleep(T_MOVE)
    iface.update_target_pose(arm, pos, wxyz_neg90, True, True)
    time.sleep(T_MOVE)

    ## same here, can delete the pre_home
    
    iface.pre_home()
    time.sleep(T_RETRACT)

    iface.home()
    time.sleep(T_HOME)
    close_grippers(iface)


def perform_push_through(poi_trace, cen_poi_vec, iface, viz=False, pin_arm=None):
    """
    Execute a push-through motion along a cable trace direction.
    poi_trace  : pixel [x, y] — push target (divergence point)
    cen_poi_vec: pixel [x, y] — push start (near the cable)
    """
    ### added a pre_home, but can be deleted if necessary
    iface.pre_home()
    time.sleep(T_RETRACT)

    iface.home()
    time.sleep(T_HOME)
    close_grippers(iface)

    place1 = get_world_coord_from_pixel_coord(cen_poi_vec, CAM_INTR)
    place2 = get_world_coord_from_pixel_coord(poi_trace,   CAM_INTR)

    if poi_trace[0] > 1904:
        arm = 'right'
        place1[1] += 0.02
        place2[1] += 0.02
        place1 = np.array([place1[0], place1[1], FOAM_DEPTH_R])
        place2 = np.array([place2[0], place2[1], FOAM_DEPTH_R])
        wxyz   = GRIP_DOWN_WXYZ_R
    else:
        arm = 'left'
        place1 = np.array([place1[0], place1[1], FOAM_DEPTH_L])
        place2 = np.array([place2[0], place2[1], FOAM_DEPTH_L])
        wxyz   = GRIP_DOWN_WXYZ_L

    ### we add 0.07 in the z before the push through (intermediate position). we can probably play around with this
    intermediate = place1 + np.array([0, 0, 0.07])

    print("PERFORMING PUSH THROUGH")
    print(place1, place2)

    _viz_target(iface, "motion/push/start", place1, wxyz)
    _viz_target(iface, "motion/push/end",   place2, wxyz)
    iface.update_target_pose(arm, intermediate, wxyz, True, True)  # above start
    time.sleep(T_MOVE)
    iface.update_target_pose(arm, place1, wxyz, True, True)         # contact at start
    time.sleep(T_CONTACT)
    iface.update_target_pose(arm, place2, wxyz, True, True)         # push to target
    time.sleep(T_CONTACT)

    retract = place2 + np.array([0, 0, 0.05])
    iface.update_target_pose(arm, retract, wxyz, True, True)
    time.sleep(T_RETRACT)

    # can be deleted
    iface.pre_home()
    time.sleep(T_RETRACT)

    iface.home()
    time.sleep(T_HOME)
    close_grippers(iface)
    return True


def perform_point_grasp_and_bin_drop_pipeline(center, angle, interface, img=None, viz=False, pin_arm=None):
    """
    Grasp an object at a pixel location with a given orientation angle,
    then drop it in the bin.
    """
    waypoint = np.array([center[0], center[1]])
    iface    = interface

    print(waypoint)
    open_grippers(iface)

    if waypoint[0] > 1750:
        arm    = 'right'
        place1 = get_world_coord_from_pixel_coord(waypoint, DEC_CAM_INTR)
        place1 = np.array([place1[0], place1[1], DEC_FIXED_DEPTH_OBJ_R])
    else:
        arm    = 'left'
        place1 = get_world_coord_from_pixel_coord(waypoint, DEC_CAM_INTR)
        place1 = np.array([place1[0], place1[1], DEC_FIXED_DEPTH_OBJ_L])

    intermediate = np.array([place1[0], place1[1], 0.26])
    wxyz = get_gripper_rot_wxyz(np.radians(angle))

    print(f"{angle=}")

    _viz_target(iface, "motion/point_grasp/grasp", place1, wxyz)
    iface.update_target_pose(arm, intermediate, wxyz, False, True)  # approach
    time.sleep(T_MOVE)
    iface.update_target_pose(arm, place1, wxyz, False, True)         # grasp depth
    time.sleep(T_CONTACT)
    close_gripper(iface, arm)

    iface.update_target_pose(arm, intermediate, wxyz, True, True)    # retract
    time.sleep(T_RETRACT)
    pipeline_drop_in_bin(iface, arm=arm)

    iface.pre_home()
    time.sleep(T_RETRACT)

    iface.home()
    time.sleep(T_HOME)


def bimanual_grasp_and_bin_drop_pipeline(center_l, angle_l, center_r, angle_r, interface, img=None, viz=False, pin_arm=None):
    """
    Both arms simultaneously grasp objects at given pixel locations and
    orientation angles, then drop them in the bin.
    """
    waypoint_l = np.array([center_l[0], center_l[1]])
    waypoint_r = np.array([center_r[0], center_r[1]])
    iface      = interface

    open_grippers(iface)

    place1_l       = get_world_coord_from_pixel_coord(waypoint_l, DEC_CAM_INTR)
    place1_l       = np.array([place1_l[0], place1_l[1], DEC_FIXED_DEPTH_OBJ_L])
    intermediate_l = np.array([place1_l[0], place1_l[1], 0.19])
    wxyz_l         = get_gripper_rot_wxyz(np.radians(angle_l))

    place1_r       = get_world_coord_from_pixel_coord(waypoint_r, DEC_CAM_INTR)
    place1_r       = np.array([place1_r[0], place1_r[1], DEC_FIXED_DEPTH_OBJ_L])
    intermediate_r = np.array([place1_r[0], place1_r[1], 0.19])
    wxyz_r         = get_gripper_rot_wxyz(np.radians(angle_r))

    _viz_target(iface, "motion/bimanual/grasp_l", place1_l, wxyz_l)
    _viz_target(iface, "motion/bimanual/grasp_r", place1_r, wxyz_r)
    # Both arms approach
    iface.update_target_pose('left',  intermediate_l, wxyz_l, False, True)
    iface.update_target_pose('right', intermediate_r, wxyz_r, False, True)
    time.sleep(T_MOVE)

    # Both arms descend to grasp depth
    iface.update_target_pose('left',  place1_l, wxyz_l, False, True)
    iface.update_target_pose('right', place1_r, wxyz_r, False, True)
    time.sleep(T_CONTACT)
    close_grippers(iface)

    # Both arms retract
    iface.update_target_pose('left',  intermediate_l, wxyz_l, True, True)
    iface.update_target_pose('right', intermediate_r, wxyz_r, True, True)
    time.sleep(T_RETRACT)

    pipeline_drop_in_bin(iface, arm='both')

    iface.pre_home()
    time.sleep(T_RETRACT)

    iface.home()
    time.sleep(T_HOME)


# ---------------------------------------------------------------------------
# get_push_coords — pure geometry, copied verbatim from motion_jacobi.py
# ---------------------------------------------------------------------------

def get_push_coords(poi_coords, img_rgb, trace_list, endpt_1, endpt_2,
                    keepout_dist=KEEPOUT_DIST, viz=False, name=""):
    """
    Algorithm for IP move planning.
    Returns (poi_coord, start_coord, vec_angle, vec_dist).
    Copied verbatim from motion_jacobi.py (pure geometry, no robot calls).
    """
    trace_1 = trace_list[endpt_1]
    trace_2 = trace_list[endpt_2]

    for p, poi_coord in enumerate(poi_coords):
        plt.close()

        close_idx_1 = get_closest_trace_idx(poi_coord, trace_1)
        close_idx_2 = get_closest_trace_idx(poi_coord, trace_2)

        poi_trace_1 = trace_1[close_idx_1]
        poi_off_1   = trace_1[close_idx_1 - 1]
        poi_trace_2 = trace_2[close_idx_2]
        poi_off_2   = trace_2[close_idx_2 - 1]

        mask = np.zeros(img_rgb.shape, dtype=np.uint8)
        for trace in trace_list:
            mask = visualize_multiple_paths(mask, [trace], [WHITE], black=True)

        poi_vec_1 = vector_direction(poi_trace_1, poi_off_1)
        poi_vec_2 = vector_direction(poi_trace_2, poi_off_2)

        vec_angle = vector_angle(poi_vec_1, poi_vec_2)
        if vec_angle > 90:
            poi_vec_2 = -poi_vec_2
            vec_angle = vector_angle(poi_vec_1, poi_vec_2)
            if vec_angle > 90:
                import pdb; pdb.set_trace()

        poi_vec_bisect = (poi_vec_1 + poi_vec_2) / 2
        bi_point       = (poi_trace_1 + poi_trace_2) / 2

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

        dilate_kernel4      = np.ones((4, 4), np.uint8)
        dilated_binary_img  = ~cv2.dilate(mask[:, :, 0], dilate_kernel4)
        dilated_binary_img[dilated_binary_img != 0] = 1

        _, labels, _, _ = cv2.connectedComponentsWithStats(dilated_binary_img)
        dist_transform  = cv2.distanceTransform(dilated_binary_img, cv2.DIST_L2, 5)

        RIDGE_THRESH = 0.0025
        ridges = frangi(dist_transform, black_ridges=False, sigmas=range(1, 5))
        ridges[ridges <  RIDGE_THRESH] = 0
        ridges[ridges >= RIDGE_THRESH] = 1

        skel_ridges = skeletonize(ridges, method="lee")
        rid0 = int(0.01 * img_rgb.shape[0])
        rid1 = int(0.01 * img_rgb.shape[1])
        skel_ridges[:rid0, :]  = 0
        skel_ridges[-rid0:, :] = 0
        skel_ridges[:, :rid1]  = 0
        skel_ridges[:, -rid1:] = 0

        ridge_points = find_points_of_interest(skel_ridges.astype(int) // 255)
        ridge_points = np.array(sorted(ridge_points, key=lambda x: vector_distance(x, poi_coord)))

        skel_ridges = remove_tri_intersections(skel_ridges)

        ITER_THRESH  = 2
        ANGLE_THRESH = 25
        DIST_THRESH  = 200

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

            dilate_kernel5      = np.ones((5, 5), np.uint8)
            dilated_skel_ridges = cv2.dilate(skel_ridges.astype(np.uint8), dilate_kernel5)

            _, labels, _, _ = cv2.connectedComponentsWithStats(dilated_skel_ridges // 255)
            closest_index   = labels[closest_point[1], closest_point[0]]
            closest_ridge   = np.where(labels == closest_index, 1, 0).astype(np.uint8)

            closest_ridge[dist_transform < keepout_dist] = 0
            skel_keepout = skeletonize(closest_ridge, method="lee")

            keepout_points = find_points_of_interest(skel_keepout.astype(int))

            if keepout_points is None or len(keepout_points) == 0:
                print("No keepout_points, moving to next ridge_point")
                continue

            start_coord = keepout_points[np.argmin(vector_distance(keepout_points, poi_coord, axis=1))]
            plt.scatter(start_coord[0], start_coord[1], s=80, c="m", zorder=10)

            poi_vec_start  = vector_direction(poi_coord, start_coord)
            poi_vec_bisect = (poi_vec_1 + poi_vec_2) / 2

            plt.axline(start_coord, slope=poi_vec_start[1] / poi_vec_start[0], color='c')
            plt.axline(poi_coord,   slope=poi_vec_bisect[1] / poi_vec_bisect[0], color='r')

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
