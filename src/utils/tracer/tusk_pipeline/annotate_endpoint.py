import numpy as np
import os
import matplotlib.pyplot as plt
import math
import cv2

input_image_folder = 'multicable_images/tier3'
output_data_folder = 'multicable_data/tier3'

glob_points = []
image = None

def get_real_world_image_points(img_path):
    global glob_points
    global image
    # resetting points
    glob_points = []
    image = cv2.imread(img_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_points_on_real_image)
    while(1):
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if k == ord('r'):
            glob_points = []
            print('Erased annotations for current image')
    cv2.destroyAllWindows()
    return glob_points

def click_points_on_real_image(event, x, y, flags, param):
    global glob_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        glob_points.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

if __name__ == '__main__':
    for i, data in enumerate(np.sort(os.listdir(input_image_folder))):
        img_path = os.path.join(input_image_folder, data)
        if data == '.DS_Store':
            continue
        test_data = plt.imread(img_path)
        np_data = (np.array(test_data)*255).astype(np.uint8)
        start_point = get_real_world_image_points(img_path) 
        data_dict = {'image': np_data, 'start_point': start_point}
        print(data_dict)
        np.save(os.path.join(output_data_folder, data.split('.')[0]), data_dict)



