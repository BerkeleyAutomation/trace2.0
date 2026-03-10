# -173.6499989, y: 0.2555691, z: 89.7842346

from untangling.point_picking import *
from autolab_core import RigidTransform
from untangling.utils.interface_rws_BRIO import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
import numpy as np
# from untangling.utils.grasp import GraspSelector
from untangling.point_picking import click_points_simple, click_points_closest
from decluttering.data.tracer.blip_pipeline.tracer import run_tracer_with_transform
import cv2
import numpy as np
# from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from untangling.utils.grasp import GraspSelector
import time
from autolab_core import RigidTransform, Point, RgbdImage, DepthImage, ColorImage
from autolab_core.transformations import rotation_matrix
from scripts.full_pipeline_trunk_BLIP import FullPipeline

from collections import OrderedDict
import argparse
import logging
import numpy as np
import colorsys
from decluttering.scripts.brio.brio_sensor import BRIOSensor

T_CAM_BASE = RigidTransform.load("/home/justinyu/multicable-decluttering/decluttering/scripts/brio/brio_to_world_bww.tf").as_frames(from_frame="brio", to_frame="base_link")

Z_OFFSET = 1.4e-3
FOAM_DEPTH = 0.0582
FOAM_DEPTH_R = 0.0582
FOAM_DEPTH_L = 0.0530

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.04-FOAM_DEPTH])
    point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
    print(point_3d_world)
    point_3d_world = point_3d_world[:3]/point_3d_world[3]
    point_3d_world[-1] = FOAM_DEPTH
    # print('non-homogenous = ', point_3d_world)
    return point_3d_world


def run_pipeline(iface):
        # iface.open_grippers()
        # iface.home()
        # iface.sync()
        cam = BRIOSensor(0)
        # print(cam.intrinsics()
        while True: 
            img = cam.read()
             
            print("Choose place points")
            place_1, _= click_points_simple(img)
            print(place_1)

            plt.scatter(place_1[0], place_1[1], c='r')
            plt.imshow(img.data)
            plt.show()


            # g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
            # place1_point = g.ij_to_point(place_1).data

            place1_point = get_world_coord_from_pixel_coord(place_1, cam.intrinsics)

            print("place1", place_1)
            print("place1_point", place1_point)

            if place1_point[1] < 0:
                place1_point[2] = FOAM_DEPTH_R
                place1_transform = RigidTransform(
                translation= place1_point,
                rotation= iface.GRIP_DOWN_R,
                from_frame=YK.r_tcp_frame,
                to_frame="base_link",
                )
                iface.go_cartesian(
                r_targets=[place1_transform],
                )
                iface.sync()
                time.sleep(0.5)
            else:
                place1_point[2] = FOAM_DEPTH_L
                place1_transform = RigidTransform(
                    translation= place1_point,
                    rotation= iface.GRIP_DOWN_R,
                    from_frame=YK.l_tcp_frame,
                    to_frame="base_link",
                )
                iface.go_cartesian(
                l_targets=[place1_transform],
                )
                iface.sync()
                time.sleep(0.5)
            iface.home()
            iface.sync()
            time.sleep(0.5)
            # iface.close_grippers()
            # iface.sync()

                
if __name__ == "__main__":
    SPEED = (0.6, 2 * np.pi)
    iface = Interface(
                ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
                speed=SPEED,
            )
    # print("cam intsinsics", iface.cam.intrinsics.fx, iface.cam.intrinsics.fy, iface.cam.intrinsics.cx, iface.cam.intrinsics.cy, iface.cam.intrinsics.skew, iface.cam.intrinsics.height, iface.cam.intrinsics.width)
    # print("proj mat", iface.cam.intrinsics.K)
    # print("frame", iface.cam.intrinsics.frame)
    # print("T_PHOXI_BASE", iface.T_PHOXI_BASE)
    iface.home()
    iface.sync()
    time.sleep(0.5)
            
    run_pipeline(iface)







 