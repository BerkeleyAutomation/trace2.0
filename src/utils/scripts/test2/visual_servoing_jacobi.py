from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform, Point
import torch 
from multiprocessing import Process, SimpleQueue, shared_memory
from decluttering.scripts.brio.brio_sensor import BRIOSensor
import numpy as np
import cv2
from decluttering.data.utils.detic_dataloader import DeticDataloader
import sys
import time
from scipy.ndimage import center_of_mass
import copy 

def find_center_of_mass(binary_mask):
    """
    Find the center of mass of pixels given a binary mask image.
    
    Parameters:
    binary_mask (numpy.ndarray): A 2D numpy array representing the binary mask image.

    Returns:
    tuple: The (row, column) coordinates of the center of mass.
    """
    # Ensure the input is a binary mask
    binary_mask = np.asarray(binary_mask, dtype=bool)
    
    # Calculate the center of mass
    center = center_of_mass(binary_mask)
    
    return center

def BRIO_process(shm_dict):
    cam = BRIOSensor(0)
    img_buffer = np.ndarray((1060, 1904, 3), dtype=np.uint8, buffer=shm_dict['image_shm'].buf)
    while True:
        # start_time = time.time()
        img = cam.read()
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)) # (1060, 1904, 3)
        img_buffer[:img.shape[0], :img.shape[1]] = img[:, :, ::-1]
        # print("FPS: {}".format(1/(time.time() - start_time)))

def det_process(shm_dict):
    img_buffer = np.ndarray((1060, 1904, 3), dtype=np.uint8, buffer=shm_dict['image_shm'].buf)
    centroid_buffer = np.ndarray((2,), dtype=np.int32, buffer=shm_dict['det_centroid_shm'].buf)
    det_img_buffer = np.ndarray((1060, 1904, 3), dtype=np.uint8, buffer=shm_dict['det_img_shm'].buf)
    detic = DeticDataloader()
    detic.create()
    detic.custom_vocab(['cup'])
    while img_buffer[0,0,0] == 0:
        time.sleep(0.1)
    while True:
        # start_time = time.time()
        # img = img_buffer.copy()
        out = detic.predict(img_buffer)
        output_im = out['vis'].get_image()
        
        # output_im = detic.visualize_output(img_buffer, out["masks"])
        # print(output_im.shape)
        det_img_buffer[:output_im.shape[0], :output_im.shape[1]] = output_im[:, :, ::-1]
        moi = None
        for idx, output in enumerate(out["class_name"]):
            # print(output)
            if output == 'cup':
                moi = out["masks"][idx]
        if moi != None:
            cm = find_center_of_mass(moi.cpu().detach().numpy())
            # print(cm)
            centroid_buffer[:] = [int(cm[1]), int(cm[2])]
        # print("FPS: {}".format(1/(time.time() - start_time)))
        
def plot_process(shm_dict):
    img_buffer = np.ndarray((1060, 1904, 3), dtype=np.uint8, buffer=shm_dict['image_shm'].buf)
    centroid_buffer = np.ndarray((2,), dtype=np.int32, buffer=shm_dict['det_centroid_shm'].buf)
    # det_img_buffer = np.ndarray((1060, 1904, 3), dtype=np.uint8, buffer=shm_dict['det_img_shm'].buf)
    while True:
        img = copy.copy(img_buffer)
        cv2.imshow("img", img)
        print(centroid_buffer)
        if int(centroid_buffer[0]) != 0 and int(centroid_buffer[1]) != 0:
            cv2.circle(img, (int(centroid_buffer[0]), int(centroid_buffer[1])), 5, (255, 0, 0), -1)
        if cv2.waitKey(1) == ord('q'):
            break
        
def create_shm(shm_dict, create=True):
    shm_dict["det_centroid_shm"] = shared_memory.SharedMemory(name="endpt_shm", create=create, size=2*32)
    shm_dict["image_shm"] = shared_memory.SharedMemory(name="image_shm", create=create, size=1060*1904*3*8)
    shm_dict["det_img_shm"] = shared_memory.SharedMemory(name="det_img_shm", create=create, size=1060*1904*3*8)
    
    return shm_dict

def create_processes(process_dict, shm_dict):
    process_dict["capture"] = Process(target=BRIO_process, args=(shm_dict,))
    process_dict["detection"] = Process(target=det_process, args=(shm_dict,))
    # process_dict["endpoints"] = Process(target=endpoint_process, args=(shm_dict,))
    process_dict["plot"] = Process(target=plot_process, args=(shm_dict,))
    # process_dict["motion"] = Process(target=motion_process, args=(shm_dict, iface))
    return process_dict

def run():
    # torch.multiprocessing.set_start_method('spawn')
    shm_dict = {}
    process_dict = {}
    
    interface = Interface(speed=0.6, file='/home/justinyu/multicable-decluttering/yumi_jacobi/starter_examples/AUTOLAB_BWW_YuMi.jacobi-project')
    interface.home()
    interface.calibrate_grippers()
    interface.open_grippers()
    
    torch.multiprocessing.set_start_method('spawn', force = True)
    try:
        shm_dict = create_shm(shm_dict)
    except:
        print("Shared memory already exists, obliterating")
        shm_dict = create_shm(shm_dict, create=False)
        for key in shm_dict:
            shm_dict[key].close()
            shm_dict[key].unlink()
    try:
        process_dict = create_processes(process_dict=process_dict, shm_dict=shm_dict)

        for key in process_dict:
            process_dict[key].start()

        # motion(shm_dict, interface)

    except Exception as e:
        print(e)
    finally:
        try:
            print("Exiting...")
            for key in process_dict:
                process_dict[key].join()
            for key in shm_dict:
                shm_dict[key].close()
                shm_dict[key].unlink()
        finally:
            sys.exit()

if __name__ == "__main__":
    run()