
import sys
sys.path.insert(0, '..')
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage, optimize as opt
from decluttering.data.tracer.tusk_pipeline.tracer import Tracer, TraceEnd
import time

GAUSSIAN_MEAN = 0 # 0
GAUSSIAN_STDDEV = 30 # 0.5
GAUSSIAN_KERNEL_SIZE = (1, 1) # (1, 1)
TRACES_DIR = '../data/traces/'

def add_gaussian_blur(img):
    blurred_img = cv2.GaussianBlur(img, GAUSSIAN_KERNEL_SIZE, cv2.BORDER_DEFAULT)
    return blurred_img


def add_gaussian_noise_greyscale(img):
    mean = GAUSSIAN_MEAN
    stddev = GAUSSIAN_STDDEV
    noise_one_d = np.zeros(img[:,:,0].shape, np.int32)
    cv2.randn(noise_one_d, mean, stddev)
    noise = np.dstack((noise_one_d, noise_one_d, noise_one_d))
    noisy_img = cv2.add(img.astype('int32'), noise)
    return noisy_img


def rotate_img(img):
    rotation_angle = random.randint(0, 360)
    # keeps same image size as original and clips edges
    img_rot = ndimage.rotate(img, rotation_angle, reshape=False)
    
    # reshapes the image size to make sure full img is included 
    # might be issue for endpoint detection model
    # img_rot = ndimage.rotate(img, rotation_angle, reshape=True)
    return img_rot


def read_img(img_path):
    img = cv2.imread(img_path)
    return img


def write_img(img, img_path):
    cv2.imwrite(img_path, img)


def write_spline(spline, spline_path):
    np.save(spline_path, spline)

def run_tracer_with_transform(img, start_pixels, endpoints, sample=True, start_time=None, viz = False):
    """
    Runs tracer. Assumes analytic tracer has already been run.
    """
    start_time = time.time()
    tracer = Tracer()
    print(f"Tracer model init time: {time.time() - start_time}")
    output  = tracer.trace(img, start_pixels, endpoints=endpoints, path_len=250, viz=viz, sample=False) # generate original (baseline) trace, do not sample

    '''
    output = {
        "trace": np.array(spline),
        "trace_end": trace_end,
        "heatmaps": heatmaps,
        "crops": crops,
        "covariances": covariances
    }
    '''
    
    return output


def _run_tracer_across_all_with_transform(num_noisy_traces):
    """
    Runs tracer with (optional) transform across all files (for testing). Assumes analytic tracer has already been run.
    """
    tracer = Tracer()

    # perform iterative DFS to recover starting pixels and corresponding images 
    path_stack = ['../data/rope_knot_images_analytic_traces']
    while path_stack:
        f_path = path_stack.pop()    
        if os.path.isdir(f_path): # if directory
            for child_name in np.sort(os.listdir(f_path)): # traverse directory, adding children to stack
                path_stack.append(os.path.join(f_path, child_name))
        elif 'starting_pixels' in f_path: # if valid file
            img_path = f_path.replace('analytic_traces', 'npy').replace('starting_pixels_', '') # recover corresponding image path
            if os.path.exists(img_path):
                start_pixels = np.load(f_path, allow_pickle=True)
                img = np.load(img_path, allow_pickle=True)

                input_id = img_path.split('img_')[1].split('.npy')[0] + "/" # retrieve input's unique identifier (input_id)
                output_folder = TRACES_DIR + input_id # make a folder for saving outputs corresponding to given input_id
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                idx = 0 # 0 = baseline, 1 and higher = noise-added
                while idx <= num_noisy_traces:
                    img_to_trace = img
                    if idx != 0:
                        img_to_trace = add_gaussian_noise_greyscale(add_gaussian_blur(img)) # only add noise if idx != 0
                    try: # valid starting points
                        # trace and save trace points, spline on image, image
                        spline, end = tracer.trace(img_to_trace, start_pixels, path_len=200, idx=idx, folder=output_folder, viz=True, sample=False)
                        write_spline(spline, output_folder + f'trace_{idx}.npy')
                        write_img(img_to_trace, output_folder + f'img_{idx}.png')
                    except: # invalid starting points
                        print(f'Invalid starting points: {f_path}')
                        os.rmdir(output_folder)
                        break
                    idx += 1
    

if __name__ == "__main__":
    if not os.path.exists(TRACES_DIR):
        os.mkdir(TRACES_DIR)
    _run_tracer_across_all_with_transform()