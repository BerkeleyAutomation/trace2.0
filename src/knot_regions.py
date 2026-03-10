import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import cm
import sys
sys.path.append('/home/justinyu/multicable-decluttering/')
from decluttering.src.run_constants import *
from decluttering.src.divergencecopy import visualize_multiple_paths


def plot(dir, viz, name):
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{name}_{dir}.png")
    if viz:
        plt.show()
    else:
        plt.close()


def find_open_knot_regions(img_down, div_point, radius=CC_RADIUS,
                           min_area=MIN_REGION_AREA, viz=False, name=""):
    """
    Inputs:
    - img_down (scaled image)
    - div_point (coordinates)
    - radius (around div_point to treat as knot)
    - min/max_area (of open region for grippers)

    Returns:
    - List[coordinates] (center of open regions)
        for grippers to enter and clean up knot
    """
    gray_img_down = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(gray_img_down, 127, 255, cv2.THRESH_BINARY_INV)

    # mask: filled white circle on black background
    mask = np.zeros_like(gray_img_down)
    mask = cv2.circle(mask, div_point.astype(int), radius, (255, 255, 255), -1)
    
    # inv_mask: filled black circle on white background
    inv_mask = cv2.bitwise_not(mask)
    black_bg = cv2.bitwise_and(img_thresh, mask)
    white_bg = cv2.bitwise_or(black_bg, inv_mask)

    ###plt###
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].imshow(gray_img_down, cmap=cm.bone)
    axs[0, 0].set_title("Grayscaled Image")

    axs[0, 1].imshow(img_thresh, cmap=cm.bone)
    axs[0, 1].set_title("Thresholded Image (rgb > 127)")
    
    axs[1, 0].imshow(black_bg, cmap=cm.bone)
    axs[1, 0].set_title("White Circle Mask && Black BG")
    
    axs[1, 1].imshow(white_bg, cmap=cm.bone)
    axs[1, 1].set_title("White Circle Mask || White BG")
    
    plt.tight_layout()
    plot("gray_masks", False, name)
    #########

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_bg)
    area = lambda i: stats[i, cv2.CC_STAT_AREA]

    component_indices = [i for i in range(1, num_labels)
                        if min_area <= area(i) <= min_area * 10]
                        # NOTE: arbitrary threshold for now

    print(f"Found {len(component_indices)} open regions")
    centroids_by_area = [centroids[i] for i in component_indices]
    
    if len(centroids_by_area) > 0:
        ###plt###
        # fig, axs = plt.subplots(2, 3, figsize=(24, 12))
        # axs = axs.flatten()

        # for i, index in enumerate(component_indices[:6]):
        #     blank_mask = np.zeros(img_thresh.shape, dtype="uint8")
        #     areas_mask = (labels == index).astype("uint8") * 255
        #     mask = cv2.bitwise_or(blank_mask, areas_mask)
            
        #     axs[i].imshow(mask, cmap=cm.bone)
        #     axs[i].scatter(div_point[0], div_point[1])
        #     axs[i].set_title(f"Area: {area(i)}")
        # plot("knot_masks", False, name)
        
        bbox = (220, 510, 780, 1070)

        plt.figure(figsize=(20, 12))
        plt.imshow(img_down[bbox[0]:bbox[1], bbox[2]:bbox[3]]) # (1060, 1904, 3)
        for i in component_indices:
            c = centroids[i]
            plt.scatter(c[0] - bbox[2], c[1] - bbox[0], s = 50, label=f"Area: {area(i)}")

        # uncomment to plot div point as a triangle on img:
        plt.scatter(div_point[0] - bbox[2], div_point[1] - bbox[0], marker='^')
        # plt.legend()
        # plot("centroids", viz, name)
        plt.axis('off')
        plt.savefig(f"{SAVE_DIR}/{name}_.png", bbox_inches='tight', pad_inches=0)
        #########
        
        return centroids_by_area[0], area(0)
    else:
        return None, None


def classify_point_in_knot(img_down, div_point, trace_list, densities_p,
                           radius=DENSITY_RADIUS, viz=False, name=""):
    
    dense_count = 0
    for traces, density in zip(trace_list, densities_p):
        for trace_pt, dense_val in zip(traces, density):
            if (dense_val > DENSITY_THRESH and
                np.linalg.norm(div_point - trace_pt) < radius):
                    dense_count += 1
                    break
    
    ###plt###
    bbox = (220, 510, 780, 1070)
    fig, ax = plt.subplots(figsize=(16, 12))
    # import pdb; pdb.set_trace()
    ax.imshow(visualize_multiple_paths(img_down, trace_list, densities_p, heat_thresh=0.5)[bbox[0]:bbox[1], bbox[2]:bbox[3]])

    # plt.title(f"Div point {div_point} (green) is INSIDE KNOT\n")
    # circle = patches.Circle(div_point, radius, linewidth=2, edgecolor="blue", facecolor="none")
    
    # ax.scatter(div_point[0], div_point[1], c="lime", s=25)
    # ax.add_patch(circle)
    # plt.tight_layout()
    # plot("knot_region", viz, name)
    plt.axis('off')
    plt.savefig(f"{SAVE_DIR}/{name}_density.png", bbox_inches='tight', pad_inches=0)
    #########
    
    if dense_count > 0:
        print("Knot!")
    else:
        print("Not a knot")
    return dense_count