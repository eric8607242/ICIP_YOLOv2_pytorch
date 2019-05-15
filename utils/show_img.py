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

    for l in range(0, label_bnd.size(0), 2):
        if label_bnd[l, 4] != 0:
            p_index = 0
            if predict_bnd[l, 4] > predict_bnd[l+1, 4]:
                p_index = l
            else:
                p_index = l+1
            x1, y1 = predict_bnd[p_index, 0] * 448, predict_bnd[p_index, 1] * 448
            x2, y2 = (predict_bnd[p_index, 0]+predict_bnd[p_index, 2])*448, (predict_bnd[p_index, 1]+predict_bnd[p_index, 3])*448
            _, predict_class = predict_bnd[p_index, -13:].max(0)

            l_x1, l_y1 = label_bnd[l, 0]*448, label_bnd[l, 1]*448
            l_x2, l_y2 = (label_bnd[l, 0]+label_bnd[l, 2])*448, (label_bnd[l, 1]+label_bnd[l, 3])*448
            _, l_class = label_bnd[l, -13:].max(0)

            draw.text((x1, y1), classes[predict_class], fill=128)
            draw.text((l_x1, l_y1), classes[l_class], fill=256)
            draw.rectangle([x1, y1, x2, y2], outline=128)
            draw.rectangle([l_x1, l_y1, l_x2, l_y2], outline=256)

    image.save("./1.png")
    

