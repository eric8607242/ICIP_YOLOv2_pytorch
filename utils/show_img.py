from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

import matplotlib.patches as patches

import torch
import torchvision.transforms.functional as F

from PIL import ImageDraw
classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower']

def show_img(image_array, predict_bnd, label_bnd):
    """
        image_array(1, 3, 224, 224)
    """
    image_size = 448
    image = F.to_pil_image(image_array)
    draw = ImageDraw.Draw(image)

    predict_bnd = predict_bnd.contiguous().view(-1, 23)
    label_bnd = label_bnd.contiguous().view(-1, 23)

    for l in range(0, label_bnd.size(0)):
        if label_bnd[l, 4] != 0:
            cell_n = l+1
            cell_x = cell_n // 7 +1
            cell_y = cell_n % 7
#             print(cell_n)
#             print(cell_x)
#             print(cell_y)
            cell_x = cell_x*(448/7)
            cell_y = cell_y*(448/7)

            p_index = 0
            if predict_bnd[l, 4] > predict_bnd[l, 9]:
                p_index = 0
            else:
                p_index = 5
#             print(predict_bnd[l])
#             print(label_bnd[l])

            width = predict_bnd[l, 2+p_index] *448
            height = predict_bnd[l, 3+p_index] *448
            center_x = cell_x+predict_bnd[l, 0+p_index] * 448
            center_y = cell_y+predict_bnd[l, 1+p_index] * 448

            x1, y1 = center_x - 0.5*width, center_y - 0.5*height
            x2, y2 = center_x + 0.5*width, center_y + 0.5*height
            _, predict_class = predict_bnd[l, -13:].max(0)

            l_width = label_bnd[l, 2] *448
            l_height = label_bnd[l, 3]*448
            l_center_x = cell_x+label_bnd[l, 0] * 448
            l_center_y = cell_y+label_bnd[l, 1] * 448

            l_x1, l_y1 = l_center_x - 0.5*l_width, l_center_y - 0.5*l_height
            l_x2, l_y2 = l_center_x + 0.5*l_width, l_center_y + 0.5*l_height
#             print(l_x1)
#             print(l_y1)
            _, l_class = label_bnd[l, -13:].max(0)

            draw.text((x1, y1), classes[predict_class], fill=128)
            draw.text((l_x1, l_y1), classes[l_class], fill=256)
            draw.rectangle([x1, y1, x2, y2], outline=128)
            draw.rectangle([l_x1, l_y1, l_x2, l_y2], outline=256)

    image.save("./1.png")

