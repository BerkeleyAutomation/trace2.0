# !!! Checked and cleaned tracing
# code begins Line ~262 and below

import os
import sys
import cv2
import copy
import time
import torch
import shutil
import colorsys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

sys.path.insert(0, '..')
use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from enum import Enum
from scipy import interpolate
from collections import OrderedDict
from imgaug.augmentables import KeypointsOnImage
from torchvision import transforms, models, utils
from decluttering.src.run_constants import DENSITY_RADIUS
from decluttering.data.tracer.model_training.config import *
from decluttering.data.tracer.model_training.src.model import KeypointsGauss
from decluttering.data.tracer.analytic_tracer import simple_uncertain_trace_single
from decluttering.data.tracer.blip_pipeline.refine_push_location import find_crossings
from scipy.stats import multivariate_normal

# Cable mask threshold: pixels with green channel below this value (normalized 0-1)
# are masked as background. Increase to filter more aggressively (try 0.5-0.65).
CABLE_MASK_THRESH = 0.65

class TraceEnd(Enum):
    EDGE = 1
    ENDPOINT = 2
    FINISHED = 3
    RETRACE = 4
    OBJECT = 5

class Tracer:

    def __init__(self) -> None:
        self.trace_config = TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp()
        if use_cuda:
            print("Autoregressive Tracer Using GPU")
            self.trace_model =  KeypointsGauss(1, img_height=self.trace_config.img_height, img_width=self.trace_config.img_width, channels=3, resnet_type=self.trace_config.resnet_type, pretrained=self.trace_config.pretrained).cuda()
            # self.trace_model.load_state_dict(torch.load('../models/tracer/tracer_model.pth')) # for remote
            self.trace_model.load_state_dict(torch.load('/home/justinyu/multicable-decluttering/decluttering/data/tracer/models/tracer/tracer_model.pth')) # for athena5
        else:
            print("Autoregressive Tracer Using CPU")
            self.trace_model =  KeypointsGauss(1, img_height=self.trace_config.img_height, img_width=self.trace_config.img_width, channels=3, resnet_type=self.trace_config.resnet_type, pretrained=self.trace_config.pretrained)
            # self.trace_model.load_state_dict(torch.load('../models/tracer/tracer_model.pth', map_location=torch.device('cpu'))) # for remote
            self.trace_model.load_state_dict(torch.load('/home/mallika/triton4-lip/cable-untangling/untangling/tracer_knot_detect/models/tracer/tracer_model.pth', map_location=torch.device('cpu'))) # for athena5
        augs = []
        augs.append(iaa.Resize({"height": self.trace_config.img_height, "width": self.trace_config.img_width}))
        self.real_img_transform = iaa.Sequential(augs, random_order=False)
        self.transform = transforms.Compose([transforms.ToTensor()])
        # TODO: fix
        self.x_buffer = 5
        self.y_buffer = 5
        self.ep_buffer = 15
        self.vit_model = None
        self.bool_detic_mask = None
        self.detic_mask = None
        
    def _get_evenly_spaced_points(self, pixels, num_points, start_idx, spacing, img_size, backward=True, randomize_spacing=True):
        pixels = np.squeeze(pixels)
        def is_in_bounds(pixel):
            return pixel[0] >= 0 and pixel[0] < img_size[0] and pixel[1] >= 0 and pixel[1] < img_size[1]
        def get_rand_spacing(spacing):
            return spacing * np.random.uniform(0.8, 1.2) if randomize_spacing else spacing
        # get evenly spaced points
        last_point = np.array(pixels[start_idx]).squeeze()
        points = [last_point]
        if not is_in_bounds(last_point):
            return np.array([])
        rand_spacing = get_rand_spacing(spacing)
        start_idx -= (int(backward) * 2 - 1)
        while start_idx > 0 and start_idx < len(pixels):
            cur_spacing = np.linalg.norm(np.array(pixels[start_idx]).squeeze() - last_point)
            if cur_spacing > rand_spacing and cur_spacing < 2*rand_spacing:
                last_point = np.array(pixels[start_idx]).squeeze()
                rand_spacing = get_rand_spacing(spacing)
                if is_in_bounds(last_point):
                    points.append(last_point)
                else:
                    points = points[-num_points:]
                    return np.array(points)[..., ::-1]
            start_idx -= (int(backward) * 2 - 1)
        points = points[-num_points:]
        return np.array(points)

    def center_pixels_on_cable(self, image, pixels):
        # for each pixel, find closest pixel on cable
        image_mask = image[:, :, 0] > 100
 
        # erode white pixels
        kernel = np.ones((2,2),np.uint8)
        image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
        white_pixels = np.argwhere(image_mask)
        
        # # visualize this
        # plt.imshow(image_mask)
        # for pixel in pixels:
        #     plt.scatter(*pixel[::-1], c='r')
        # plt.show()

        processed_pixels = []
        for pixel in pixels:
            # find closest pixel on cable
            # print(pixel)
            distances = np.linalg.norm(white_pixels - pixel, axis=1)
            closest_pixel = white_pixels[np.argmin(distances)]
            processed_pixels.append([closest_pixel])
        return np.array(processed_pixels)

    def call_img_transform(self, img, kpts):
        img = img.copy()
        normalize = False
        if np.max(img) <= 1.0:
            normalize = True
        if normalize:
            img = (img * 255.0).astype(np.uint8)
        img, keypoints = self.real_img_transform(image=img, keypoints=kpts)
        if normalize:
            img = (img / 255.0).astype(np.float32)
        return img, keypoints

    def draw_spline(self, crop, x, y, label=False):
        # x, y = points[:, 0], points[:, 1]
        if len(x) < 2:
            raise Exception("if drawing spline, must have 2 points minimum for label")
        # x = list(OrderedDict.fromkeys(x))
        # y = list(OrderedDict.fromkeys(y))
        tmp = OrderedDict()
        for point in zip(x, y):
            tmp.setdefault(point[:2], point)
        mypoints = np.array(list(tmp.values()))
        x, y = mypoints[:, 0], mypoints[:, 1]
        k = len(x) - 1 if len(x) < 4 else 3
        if k == 0:
            x = np.append(x, np.array([x[0]]))
            y = np.append(y, np.array([y[0] + 1]))
            k = 1

        tck, u = interpolate.splprep([x, y], s=0, k=k)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
        xnew = np.array(xnew, dtype=int)
        ynew = np.array(ynew, dtype=int)

        x_in = np.where(xnew < crop.shape[0])
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        x_in = np.where(xnew >= 0)
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        y_in = np.where(ynew < crop.shape[1])
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]
        y_in = np.where(ynew >= 0)
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]

        spline = np.zeros(crop.shape[:2])
        if label:
            weights = np.ones(len(xnew))
        else:
            weights = np.geomspace(0.5, 1, len(xnew))

        spline[xnew, ynew] = weights
        spline = np.expand_dims(spline, axis=2)
        spline = np.tile(spline, 3)
        spline_dilated = cv2.dilate(spline, np.ones((3,3), np.uint8), iterations=1)
        return spline_dilated[:, :, 0]

    def get_crop_and_cond_pixels(self, img, condition_pixels, center_around_last=False):
        w = self.trace_config.crop_width
        center_of_crop = condition_pixels[-self.trace_config.pred_len*(1 - int(center_around_last))-1]

        img = np.pad(img, ((w, w), (w, w), (0, 0)), 'constant')
        center_of_crop = center_of_crop.copy() + w

        crop = img[max(0, center_of_crop[0] - w): min(img.shape[0], center_of_crop[0] + w + 1),
                   max(0, center_of_crop[1] - w): min(img.shape[1], center_of_crop[1] + w + 1)]
        img = crop
        top_left = [center_of_crop[0] - w, center_of_crop[1] - w]
        condition_pixels = [[pixel[0] - top_left[0] + w, pixel[1] - top_left[1] + w]
                            for pixel in condition_pixels]

        return img, np.array(condition_pixels)[:, ::-1], top_left

    def get_trp_model_input(self, crop, crop_points, center_around_last=False):
        # print("Number of points before processing:", len(crop_points))

        kpts = KeypointsOnImage.from_xy_array(crop_points, shape=crop.shape)
        # print("Number of kpts from_xy_array:", len(kpts))

        img, kpts = self.call_img_transform(img=crop, kpts=kpts)
        # print("Number of kpts call_img_transform:", len(kpts))
        points = []
        for k in kpts:
            points.append([k.x,k.y])
        points = np.array(points)
        
        # print("Number of points after processing:", len(points))

        points_in_image = []
        for i, point in enumerate(points):
            px, py = int(point[0]), int(point[1])
            if px not in range(img.shape[1]) or py not in range(img.shape[0]):
                continue
            points_in_image.append(point)
        points = np.array(points_in_image)
        # print("Number of points after more processing:", len(points))
        angle = 0
        if self.trace_config.rot_cond:
            # Degenerate transformed keypoints can leave <2 valid points in crop.
            # In that case, skip rotation instead of crashing tracing.
            if len(points) >= 2:
                if center_around_last:
                    dir_vec = points[-1] - points[-2]
                elif len(points) >= self.trace_config.pred_len + 2:
                    dir_vec = points[-self.trace_config.pred_len-1] - points[-self.trace_config.pred_len-2]
                else:
                    dir_vec = None

                if dir_vec is not None:
                    dir_norm = np.linalg.norm(dir_vec)
                    if np.isfinite(dir_norm) and dir_norm > 1e-8:
                        angle = np.arctan2(dir_vec[1], dir_vec[0])

            # rotate image specific angle using cv2.rotate
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle*180/np.pi, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


        # rotate all points by angle around center of image
        points = points - np.array([img.shape[1]/2, img.shape[0]/2])
        points = np.matmul(points, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        points = points + np.array([img.shape[1]/2, img.shape[0]/2])

        # Draw condition spline only when we have enough valid points.
        img[:, :, 0] = 0
        if center_around_last:
            spline_pts = points
        else:
            spline_pts = points[:-self.trace_config.pred_len]

        if len(spline_pts) >= 2:
            img[:, :, 0] = self.draw_spline(img, spline_pts[:, 1], spline_pts[:, 0])  # * cable_mask

        cable_mask = np.ones(img.shape[:2])
        cable_mask[img[:, :, 1] < CABLE_MASK_THRESH] = 0

        if use_cuda:
            return self.transform(img.copy()).cuda(), points, cable_mask, angle
        else:
            return self.transform(img.copy()), points, cable_mask, angle

    def _is_uncovered_area_touching_before_idx(self, image, points, idx, endpoints):
        if idx is None or endpoints is None:
            return False
        image = image.copy()
        image[650:] = 0.0
        bs = 22
        for endpoint in endpoints:
            image[endpoint[0] - bs: endpoint[0]+bs, endpoint[1] - bs:endpoint[1] + bs] = 0

        uncovered_pixels = self._uncovered_pixels(image, points)
        if len(uncovered_pixels) < 30:
            return False
        image_draw = image.copy()
        for i in range(idx, len(points) - 1):
            cv2.line(image_draw, tuple(points[i])[::-1], tuple(points[i+1])[::-1], 0, 10)
        image_draw_mask = ((image_draw > 100) * 255).astype(np.uint8)

        _, labels, _, _ = cv2.connectedComponentsWithStats(image_draw_mask[..., 0], connectivity=8)
        uncovered_pixel_components = labels[uncovered_pixels[:, 0], uncovered_pixels[:, 1]]
        points_components = labels[points[:idx, 0], points[:idx, 1]]
        difference_matrix = uncovered_pixel_components[:, None] - points_components[None, ...]
        return np.sum(difference_matrix == 0) < 10


    # !!! Checked and cleaned tracing code below

    def get_distance_cumsum(self, lst):
        distances = np.linalg.norm(lst[1:] - lst[:-1], axis=1)
        cumsum = np.concatenate(([0], np.cumsum(distances)))
        return cumsum[-1] / 1000


    def weighted_cov_matrix(self, data):
        rows, cols = data.shape
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        data_f = data.flatten()
        
        weight = np.sum(data_f)
        if (not np.isfinite(weight)) or weight <= 1e-8:
            # Fallback to a small isotropic covariance at image center when heatmap is degenerate.
            return np.eye(2, dtype=float), cols / 2.0, rows / 2.0

        mean_x = np.sum(x_flat * data_f) / weight
        mean_y = np.sum(y_flat * data_f) / weight
        if (not np.isfinite(mean_x)) or (not np.isfinite(mean_y)):
            return np.eye(2, dtype=float), cols / 2.0, rows / 2.0
        
        x_cent = x_flat - mean_x
        y_cent = y_flat - mean_y
        
        cov_xx = np.sum(data_f * x_cent * x_cent) / weight
        cov_xy = np.sum(data_f * x_cent * y_cent) / weight
        cov_yy = np.sum(data_f * y_cent * y_cent) / weight
        
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        if not np.isfinite(cov).all():
            cov = np.eye(2, dtype=float)
        return cov, mean_x, mean_y


    def _trace(self, img, start_pts, path_len, endpoints=None, radius=DENSITY_RADIUS, use_vit=False):
        
        path = [sp for sp in start_pts]
        num_condition_pts = self.trace_config.condition_len
        heatmaps, crops, covariances, vit, density = [], [], [], [], []

        for _ in range(path_len):
            global_yx = None
            condition_pixels = [p for p in path[-num_condition_pts:]]
            
            crop, cond_pixels_in_crop, top_left = self.get_crop_and_cond_pixels(
                img, condition_pixels, center_around_last=True)
            # import pdb; pdb.set_trace()
            if len(cond_pixels_in_crop) < num_condition_pts:
                raise Exception(f"cond_pixels ({len(cond_pixels_in_crop)}) must be >= {num_condition_pts}")
            
            ymin, xmin = np.array(top_left) - self.trace_config.crop_width

            if use_vit:
                vit_img_transforms = transforms.Compose([
                    # transforms.ToTensor(),
                    # transforms.ConvertImageDtype(torch.float64),
                    transforms.ToPILImage(mode='RGB'),
                    transforms.Resize((224, 224)), # resize the 65*65 crop to 224 * 224
                    transforms.ToTensor(),         # convert image to pytorch tensors
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                with torch.no_grad():
                    batch = []
                    # normalized_img_crop = np.zeros((800, 800))
                    # normalized_img_crop = cv2.normalize(crop, normalized_img_crop, 0, 255, cv2.NORM_MINMAX)
                    # cv2.imwrite("/home/justinyu/vsumedh/tmp/vit-non-transformed.png", normalized_img_crop)
                    # utils.save_image(torch.from_numpy(crop), "/home/justinyu/vsumedh/tmp/vit-untransformed.png")
                    # crop = Image.fromarray(np.uint8(crop)*255)
                    
                    crop_vit_input = copy.deepcopy(crop)
                    crop_modified = (crop_vit_input + 1) * 127.5
                    crop_modified = crop_modified.astype(np.uint8)
                    transformed_img_crop = vit_img_transforms(crop_modified)

                    # normalized_trans_img_crop = cv2.normalize(crop, transformed_img_crop, 0, 255, cv2.NORM_MINMAX)
                    # plt.imsave("/home/justinyu/vsumedh/tmp/vit.png", crop)
                    # Image.fromarray(transformed_img_crop).save("/home/justinyu/vsumedh/tmp/vit.png")
                    # utils.save_image(normalized_trans_img_crop, "/home/justinyu/vsumedh/tmp/vit.png")
                    # utils.save_image(normalized_trans_img_crop, "/home/justinyu/vsumedh/tmp/vit.png")
                    # utils.save_image(transformed_img_crop, "/home/justinyu/vsumedh/tmp/vit.png")

                    batch.append(transformed_img_crop)
                    outs = self.vit_model(torch.stack(batch).to('cuda'))
                    probs = nn.functional.softmax(outs, dim=-1)
                    yes_conf_normalized_value = probs[0][1].item()

                # print(f"Elapsed time for ViT (single crop): {time.time() - start_time}")
                # append normalized [0.0, 1.0] value to 'vit'
                vit.append(yes_conf_normalized_value)
            ###########
            
            model_input, _, cable_mask, angle = self.get_trp_model_input(
                crop, cond_pixels_in_crop, center_around_last=True)

            crop_eroded = cv2.erode((cable_mask).astype(np.uint8), np.ones((2, 2)), iterations=1)

            model_output = self.trace_model(model_input.unsqueeze(0)).detach().cpu().numpy().squeeze()
            model_output *= crop_eroded.squeeze()
            model_output = cv2.resize(model_output, (crop.shape[1], crop.shape[0]))

            # undo rotation if done in preprocessing
            mdim = model_output.shape

            M = cv2.getRotationMatrix2D((mdim[1]/2, mdim[0]/2), -angle*180/np.pi, 1)
            model_output = cv2.warpAffine(model_output, M, (mdim[1], mdim[0]))
            
            cov, mean_x, mean_y = self.weighted_cov_matrix(model_output)
            cov = np.array(cov, dtype=float)
            # Ensure covariance is finite and PSD for scipy.stats.multivariate_normal.
            if cov.shape != (2, 2) or not np.isfinite(cov).all():
                cov = np.eye(2, dtype=float)
            cov = 0.5 * (cov + cov.T)
            cov += np.eye(2, dtype=float) * 1e-6
            if (not np.isfinite(mean_x)) or (not np.isfinite(mean_y)):
                mean_x, mean_y = mdim[1] / 2.0, mdim[0] / 2.0
            rv = multivariate_normal((mean_x, mean_y), cov, allow_singular=True)

            heatmaps.append(model_output)
            #  crops.append(something)???
            covariances.append(rv.entropy())

            ### DENSE CODE ###
            crop_bin = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            w, h = crop_bin.shape
            x, y = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))

            rad_mask = ((x - w//2)**2 + (y - h//2)**2) < (radius ** 2)
            curr_sum = (crop_bin * rad_mask).sum()
            density.append(curr_sum)
            ### DENSE CODE ###

            yx = np.unravel_index(model_output.argmax(), mdim)
            # yx *= np.array([crop.shape[0] / config.img_height, crop.shape[1] / config.img_width])
            global_yx = np.array([yx[0] + ymin, yx[1] + xmin]).astype(int)
            
            # No idea why we have to cast path into list despite pdb saying it is. Errors otherwise
            path = list(path)
            path.append(global_yx)
            
            assert self.bool_detic_mask is not None

            is_intersecting_mask = self.bool_detic_mask[global_yx[0]*2][global_yx[1]*2]
            if is_intersecting_mask:
                # import pdb; pdb.set_trace()
                # plt.imshow(self.bool_detic_mask.cpu().numpy())
                # plt.show()
                # plt.imshow(self.detic_mask.cpu().numpy())
                # plt.scatter(global_yx[1]*2, global_yx[0]*2)
                # plt.show()
                return path, TraceEnd.OBJECT, heatmaps, crops, covariances, vit, None, density, self.detic_mask[global_yx[0]*2, global_yx[1]*2]

            # If trace did a near 180, check for object
            if len(path) > 2:
                vec_1 = path[-2:]
                vec_2 = path[-3:-1]
                vec_1 = vec_1[1] - vec_1[0]
                vec_2 = vec_2[1] - vec_2[0]
                n1 = np.linalg.norm(vec_1)
                n2 = np.linalg.norm(vec_2)
                if n1 > 1e-8 and n2 > 1e-8 and np.isfinite(n1) and np.isfinite(n2):
                    vec_1 = vec_1 / n1
                    vec_2 = vec_2 / n2
                else:
                    vec_1 = None
                    vec_2 = None
                if vec_1 is not None and vec_2 is not None and np.dot(vec_1, vec_2) < -0.3:
                    # Get the point where the turn happened
                    turn_point = path[-2]
                    
                    # Check a small region around the turn point for objects
                    search_radius = 30  # pixels
                    y_min = max(0, turn_point[0]*2 - search_radius)
                    y_max = min(self.bool_detic_mask.shape[0], turn_point[0]*2 + search_radius)
                    x_min = max(0, turn_point[1]*2 - search_radius)
                    x_max = min(self.bool_detic_mask.shape[1], turn_point[1]*2 + search_radius)
                    
                    # If there's an object in this region, end trace with OBJECT
                    region = self.bool_detic_mask[y_min:y_max, x_min:x_max]
                    if region.any():
                        # Find the object number from detic_mask
                        object_nums = self.detic_mask[y_min:y_max, x_min:x_max][region.to(bool)]
                        most_common_object = torch.mode(object_nums)[0]
                        return path, TraceEnd.OBJECT, heatmaps, crops, covariances, vit, None, density, most_common_object
                
            if (global_yx[0] > (img.shape[0] - self.y_buffer) or
                global_yx[0] < self.y_buffer or
                global_yx[1] > (img.shape[1] - self.x_buffer) or
                global_yx[1] < self.x_buffer):

                return path, TraceEnd.EDGE, heatmaps, crops, covariances, vit, None, density, -1

            if endpoints is not None:
                for endpt in endpoints:
                    pix_dist = self.get_distance_cumsum(np.array(path))

                    if ((abs(global_yx[0] - endpt[0])) < self.ep_buffer and
                        (abs(global_yx[1] - endpt[1])) < self.ep_buffer and
                        pix_dist > 0.8):
                        
                        # max_sums = find_crossings(img, path)
                        return path, TraceEnd.ENDPOINT, heatmaps, crops, covariances, vit, endpt, density, -1
            
            if len(path) > 20:
                path = np.array(path)
                for i in range(len(path) - 20):

                    next = path[i:i+10]
                    last = path[-10:]
                    diff = np.linalg.norm(next - last)
                    diffrev = np.linalg.norm(next - last[::-1])

                    if diff < 10 or diffrev < 30:
                        # max_sums = find_crossings(img, path)
                        return path, TraceEnd.RETRACE, heatmaps, crops, covariances, vit, None, density, -1
        ###########

        # max_sums = find_crossings(img, path)
        return path, TraceEnd.FINISHED, heatmaps, crops, covariances, vit, None, density, -1


    def trace(self, img, start_pts, endpoints=None, path_len=20, use_vit=False):
        """
        Assumes analytic tracer has been
            run (to obtain start_pts)
        Returns: {
            "trace": np.array(spline),
            "trace_end": TRACE_END enum,
            heatmaps, crops, covariances
            vit, endpoint, density
        }
        """
        # pixels = self.center_pixels_on_cable(img, start_pts)
        # for pixel in pixels:
        #     p = pixel[0]
        #     if (p[0] >= 0 and p[1] >= 0 and
        #         p[0] < img.shape[0] and p[1] < img.shape[1]):
        #             break

        if img.max() > 1:
            img = (img / 255.0).astype(np.float32)

        num_condition_pts = self.trace_config.condition_len
        if start_pts is None or len(start_pts) < num_condition_pts:
            raise Exception(f"start_pts ({len(start_pts)}) must be >= {num_condition_pts}")
        else:
            print(f"Beginning trace with {len(start_pts)} start pixels")

        start_time = time.time()
        spline, trace_end, heatmaps, crops, covariances, vit, t_endpoint, densities, mask_num = self._trace(
            img, start_pts, path_len, endpoints=endpoints, use_vit=use_vit)
        print(f"Elapsed time for tracing: {time.time() - start_time:.2f}s")

        covariances = np.concatenate(
            (np.ones(len(start_pts)) * min(covariances), covariances), axis=0)
        
        vit = np.concatenate((np.zeros(len(start_pts)), vit), axis=0)
        
        output = {
            "trace": np.array(spline),
            "trace_end": trace_end,
            "heatmaps": heatmaps,
            "crops": crops,
            "covariances": covariances,
            "vit": vit,
            "t_endpoint": t_endpoint,
            "densities": densities,
            "mask_num": mask_num
        }
        return output


    def visualize_path(self, img, path, color=None, black=False):

        def color_heatmap(color, i, path):
            if color is not None:
                pct = (1 - np.array(color)[i]) / 3
            else:
                pct = i / len(path)
            pct1 = colorsys.hsv_to_rgb(pct, 1, 1)
            return [idx * 255 for idx in pct1][:3]

        img = img.copy()
        width = 2 if not black else 5
        
        for i in range(len(path) - 1):
            if not isinstance(path, OrderedDict):
                pt1 = tuple(path[i].astype(int))
                pt2 = tuple(path[i + 1].astype(int))
            else:
                path_keys = list(path.keys())
                pt1 = path_keys[i]
                pt2 = path_keys[i + 1]
            
            cv2.line(img, pt1[::-1], pt2[::-1], color_heatmap(color, i, path), width)
        return img


class AnalyticTracer(Tracer):

    def trace(self, img, start_pts, endpoints=None, path_len=20, viz=False):
        # pixels = self.center_pixels_on_cable(img, start_pts)

        # for pixel in pixels:
        #     p = pixel[0]
        #     if (p[0] >= 0 and p[1] >= 0 and
        #         p[0] < img.shape[0] and p[1] < img.shape[1]):
        #             break

        # starting_pts = self._get_evenly_spaced_points(pixels, self.trace_config.condition_len, start_idx, 
        #     self.trace_config.cond_point_dist_px, img.shape, backward=False, randomize_spacing=False)
        
        spline, trace_end = simple_uncertain_trace_single.trace(
            img, start_pts, None, exact_path_len=path_len, endpoints=endpoints)
        
        if spline is None:
            spline = start_pts
        return np.array(spline).astype(int), trace_end


if __name__ == '__main__':
    trace_test = './trace_test'
    if os.path.exists(trace_test):
        shutil.rmtree(trace_test)
    os.mkdir(trace_test)

    tracer = Tracer()
    analytic_tracer = AnalyticTracer()
    eval_folder = '../data/real_data/real_data_for_tracer/test'
    for i, data in enumerate(np.sort(os.listdir(eval_folder))):
        if i == 0 or i == 19:
            continue
        test_data = np.load(os.path.join(eval_folder, data), allow_pickle=True).item()
        img = test_data['img']
        img_cp = img.copy()
        img[-130:, ...] = 0
        plt.imsave(f'./binary_mask_experiments/trace_{i}_img.png', img)
        thresh_img = np.where(img[:,:,:3] > 100, 255, 0).astype('uint8')
        start_pixels = np.array(test_data['pixels'][0], dtype=np.uint32)[::-1]
        print(start_pixels)
        start_pixels, _ = analytic_tracer.trace(thresh_img, start_pixels, path_len=6, viz=False, idx=i)
        if len(start_pixels) < 5:
            continue
        # spline = tracer.trace(img_cp, start_pixels, path_len=200, viz=True, idx=i)
        spline = tracer.trace(img, start_pixels, path_len=200, viz=True, idx=i, mask=False, folder="binary_mask_experiments")
        spline = tracer.trace(thresh_img, start_pixels, path_len=200, viz=True, idx=i, mask=True, folder="binary_mask_experiments")
