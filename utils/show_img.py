from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

import matplotlib.patches as patches

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import ImageDraw
classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower']

def show_img(image_array, predict_bnd, label_bnd, anchor_box, S=7, B=2):
    """
    """
    anchor_box = torch.FloatTensor(anchor_box)
    image_size = 448
    image = TF.to_pil_image(image_array.byte())
    draw = ImageDraw.Draw(image)

    predict_bnd = predict_bnd.contiguous().view(-1, 38)
    predict_cls = predict_bnd[:, 25:].view(-1, 13)
    predict_bnd = predict_bnd[:, :25].view(-1, B, 5)
    predict_bnd[:, :, :2] = predict_bnd[:, :, :2].sigmoid()
    predict_bnd[:, :, 2:4] = (predict_bnd[:, :, 2:4].sigmoid()*10).exp() * anchor_box
    predict_bnd[:, :, 4] = predict_bnd[:, :, 4].sigmoid()
    predict_cls = F.softmax(predict_cls, dim=-1)

    label_cls = label_bnd[:, :, 25:].view(-1, 13)
    label_bnd = label_bnd[:, :, :25].view(-1, B, 5)

    for l in range(0, label_bnd.size(0)):
        for b in range(B):
            if predict_bnd[l, b, 4] >= 0.6:
                cell_n = l+1
                cell_x = cell_n // S +1
                cell_y = cell_n % S
                
                cell_x = cell_x*(448/S)
                cell_y = cell_y*(448/S)

                p_index = 0
                p_value = 0

                width = predict_bnd[l, b, 2+p_index] *448
                height = predict_bnd[l, b, 3+p_index] *448
                center_x = cell_x+predict_bnd[l, b, 0+p_index] * 32
                center_y = cell_y+predict_bnd[l, b, 1+p_index] * 32

                x1, y1 = center_x - 0.5*width, center_y - 0.5*height
                x2, y2 = center_x + 0.5*width, center_y + 0.5*height
                _, predict_class = predict_cls[l, :].max(0)

                l_width = label_bnd[l, b, 2] *448
                l_height = label_bnd[l, b,  3]*448
                l_center_x = cell_x+label_bnd[l, b, 0] * 32
                l_center_y = cell_y+label_bnd[l, b, 1] * 32

                l_x1, l_y1 = l_center_x - 0.5*l_width, l_center_y - 0.5*l_height
                l_x2, l_y2 = l_center_x + 0.5*l_width, l_center_y + 0.5*l_height
                
                _, l_class = label_cls[l, :].max(0)

                draw.text((x1, y1), classes[predict_class]+str(predict_bnd[l, b, 4].item()), fill=128)
                draw.rectangle([x1, y1, x2, y2], outline=128)
                if label_bnd[l, b, 4] == 1:
                    draw.text((x1, y1), classes[predict_class]+str(predict_bnd[l, b, 4].item()), fill=(255, 204, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 204, 0), width=5)
                    draw.text((l_x1, l_y1), classes[l_class], fill=256)
                    draw.rectangle([l_x1, l_y1, l_x2, l_y2], outline=256, width=5)
                else:
                    if predict_cls[l, predict_class] > 0.8:
                        draw.rectangle([x1, y1, x2, y2], outline=128)
                        draw.text((x1, y1), classes[predict_class]+str(predict_bnd[l, b, 4].item()), fill=128)

    image.save("./1.png")

