from decluttering.deprecated.demopipeline import DemoPipeline
from autolab_core import RigidTransform, Point
from multiprocessing import Process, SimpleQueue, shared_memory
import numpy as np
from untangling.utils.tcps import ABB_WHITE
from untangling.utils.interface_rws import Interface
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
import sys 
import time

def motion(iface):
    # iface = Interface(
    #     "1703005",
    #     ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
    #     ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
    #     speed=SPEED,
    # )


    # print(iface.get_FK('left'))
    # print(iface.get_FK('right'))
    iface.home()

    # iface.calibrate_grippers()
    # iface.open_grippers()


    wp1_l = RigidTransform(
        rotation=iface.GRIP_DOWN_R,
        translation=[0.4, 0.3, 0.2],
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    # wp2_l = RigidTransform(
    #     rotation=[[-1.0000000, -0.0000000,  0.0000000], 
    #                 [-0.0000000,  0.9659258, -0.2588190], 
    #                 [-0.0000000, -0.2588190, -0.9659258]],
    #     translation=[0.4, 0.4, 0.1]
    # )
    wp3_l = RigidTransform(
        rotation=iface.GRIP_DOWN_R,
        translation=[0.3, 0.3, 0.15],
        from_frame=YK.l_tcp_frame,
        to_frame="base_link",
    )
    wp1_r = RigidTransform(
        rotation=iface.GRIP_DOWN_R,
        translation=[0.3, -0.2, 0.15],
        from_frame=YK.r_tcp_frame,
        to_frame="base_link",
    )
    wp2_r = RigidTransform(
        rotation=iface.GRIP_DOWN_R,
        translation=[0.35, -0.15, 0.2],
        from_frame=YK.r_tcp_frame,
        to_frame="base_link",
    )
    # wp3_r = RigidTransform(
    #     rotation=[[-0.7071068,  0.7071068,  0.0000000], 
    #                 [0.5000000,  0.5000000,  0.7071068], 
    #                 [0.5000000,  0.5000000, -0.7071068]],
    #     translation=[0.45, -0.08, 0.15]
    # )

    # iface.go_cartesian(l_targets = [wp1_l], r_targets=[wp1_r])
    # iface.sync()
    # iface.go_linear_single(l_target=wp1_l, r_target=wp1_r)
    # iface.go_linear_single(l_target=wp2_l, r_target=wp2_r)
    # iface.go_linear_single(l_target=wp3_l, r_target=wp3_r)

    iface.home()
    iface.sync()
    iface.go_delta([0, 0, -0.1])
    iface.sync()
    iface.go_delta([0, -0.1, 0], [0, 0.1, 0])
    iface.sync()
    iface.go_delta([0.1, -0.1, 0], [0.1, 0.1, 0])
    iface.sync()
    iface.home()
    iface.sync()

def listener_process(iface):
    while True:
        time.sleep(0.05)
        try:
            with iface.y._lock:
                if iface.y.left._iface.services().main().is_stationary('ROB_L') or iface.y.right._iface.services().main().is_stationary('ROB_R'):
                    pass
                elif not iface.y.left._iface.services().main().is_stationary('ROB_L') or not iface.y.right._iface.services().main().is_stationary('ROB_R'):
                    print("Arms moving at time: ", time.time())
        except RuntimeError:
            pass

def create_processes(process_dict, iface):
    
    process_dict["plot"] = Process(target=listener_process, args=(iface,))
    return process_dict

def run():
    process_dict = {}
    SPEED = (0.4, 6 * np.pi) #(0.6, 6 * np.pi)

    iface = Interface(
    "1703005",
    ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
    ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame),
    speed=SPEED,
    )

    try:
        process_dict = create_processes(process_dict=process_dict, iface=iface)

        for key in process_dict:
            process_dict[key].start()

        motion(iface)

    except Exception as e:
        print(e)
    finally:
        try:
            print("Exiting...")
            for key in process_dict:
                process_dict[key].join()
            # for key in shm_dict:
            #     shm_dict[key].close()
            #     shm_dict[key].unlink()
        finally:
            sys.exit()

if __name__ == "__main__":
    run()