# Based heavily on: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

import time
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from decluttering.data.detectron2_repo.train import get_config
setup_logger()


class EndpointDataloader():

    def __init__(self, use_hub_detect=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = get_config()

        if use_hub_detect:
            from detectron2 import model_zoo
            self.thresh = 0.90
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            # Trained on 100 images
            # cfg.MODEL.WEIGHTS = "/home/justinyu/vsumedh/endpt-detector/src/output/model_final.pth"
            # Trained on 500 images
            cfg.MODEL.WEIGHTS = "/home/justinyu/vsumedh/endpt-detector/src/output-new/model_final.pth"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thresh
        else:
            self.thresh = 0.99
            cfg.MODEL.WEIGHTS = "/home/justinyu/multicable-decluttering/decluttering/data/detectron2_repo/models/endpoint_model.pth"
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thresh
        self.predictor = DefaultPredictor(cfg)


    def predict(self, image):
        st = time.time()
        outputs = self.predictor(image)
        end_boxes = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()

        endpoints = []
        for box in end_boxes:
            xmin, ymin, xmax, ymax = box
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            endpoints.append([y, x])
        
        endpoints = np.array(endpoints).astype(np.int32)
        output = {
            "boxes": end_boxes,
            "endpoints": endpoints
        }
        return output