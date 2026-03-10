import matplotlib.pyplot as plt

from phoxipy import PhoXiSensor

p = PhoXiSensor("1703005")
p.start()

frame = p.read()
plt.title("Intensity Image")
plt.imshow(frame.color.data)
plt.tight_layout()
plt.show()

p.stop()

img = frame.color.data
from decluttering.data.detectron2_repo import analysis as loop_detectron
import numpy as np 
import time

def get_endpoints(img):
    # model not used, already specified in loop_detectron
    # self.img = self.iface.take_image()
    endpoint_start = time.time()
    endpoint_boxes, out_viz = loop_detectron.predict(img, thresh=0.99, endpoints=True)
    endpoint_end = time.time()
    print("Endpoint detection took {} seconds".format(endpoint_end - endpoint_start))
    plt.clf()
    plt.imshow(out_viz)
    plt.title("Endpoints detected")
    plt.show()
    plt.imsave("/home/justinyu/multicable-decluttering/misc_images/endpoints_detected.png", out_viz)

get_endpoints(img)