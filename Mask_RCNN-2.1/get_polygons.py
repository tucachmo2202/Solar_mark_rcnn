import cv2
import os
import json 
import numpy as np
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display
import cv2
import os
import json 
from datetime import datetime

#Hàm này sinh màu ngẫu nhiên màu cho các mặt nạ
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

#Hàm áp mặt nạ vào trong ảnh
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

save_folder = "/home/manhas/Desktop/result_image/save_folder/"

file_names = os.listdir(save_folder)
file_names = [f for f in file_names if f.endswith(".json")]
black = cv2.imread("black.jpg") #Black image 1024*1024
for file_name in file_names:
    result = json.load(open("save_folder" + file_name))
    mask = result["masks"]
    info = result["info"]
    try:
        mask = np.asarray(mask)
        N = mask.shape[2]
        print(N)
        colors = random_colors(N)
        boxes = []
        for i in range(N):
            black_cp = black.copy()
            color = colors[i]
            mas = mask[:, :, i]
            masked_image = apply_mask(black_cp, mas, color)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            #Tìm contours
            ret,thresh = cv2.threshold(masked_image,1,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            cnt = contours[0]
            #Tìm hình chữ nhật bao contours
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.tolist()
            boxes.append(box)
        print(len(boxes))
    except:
        boxes = []
    mask = mask.tolist()
    results = {"boxes": boxes, "info": info}
    json.dump(results, open("./box/" + file_name.split("jpg")[0] + ".json", "w"))
