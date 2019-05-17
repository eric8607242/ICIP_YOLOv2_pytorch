import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

import xml.etree.ElementTree as ET
import xmltodict

import csv
import PIL.Image

from torch.utils.data import Dataset, DataLoader

classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower']

class ICIPDetectionset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_size, S=7, B=2, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.transform = transform
        
        self.S = S
        self.B = B

        self.images_path = [join(self.image_dir, f) for f in listdir(self.image_dir) if isfile(join(self.image_dir, f))]
        self.annotation_path = [join(self.annotation_dir, f) for f in listdir(self.annotation_dir) if isfile(join(self.annotation_dir, f))]

        self.images_path.sort()
        self.annotation_path.sort()

    def _xml_parser(self, xml_path):
        xml_data = None
        with open(xml_path) as fd:
            xml_data = xmltodict.parse(fd.read())
        return xml_data["annotation"]

    def _image_processing(self, img_path, xml_data):
        width = xml_data["size"]["width"]
        height = xml_data["size"]["height"]

        image = PIL.Image.open(img_path)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), PIL.Image.ANTIALIAS)

        ratio_x = self.image_size / int(width)
        ratio_y = self.image_size/ int(height)

        return image, ratio_x, ratio_y

    def _get_reat_axis(self, bndbox, ratio_x, ratio_y):
        xmin = int(bndbox["xmin"])*ratio_x
        xmax = int(bndbox["xmax"])*ratio_x
        ymin = int(bndbox["ymin"])*ratio_y
        ymax = int(bndbox["ymax"])*ratio_y

        width = xmax - xmin
        height = ymax - ymin

        center_x = int(width/2 + xmin)
        center_y = int(height/2 + ymin)

        cell_size = self.image_size / self.S

        cell_x = center_x // cell_size 
        cell_y = center_y // cell_size 

        relative_x = center_x - cell_x*cell_size
        relative_y = center_y - cell_y*cell_size
        
        # print("xmin %f, xmax %f, center_x %f, cell_x %f, relative %f, width %f" % (xmin, xmax, center_x, cell_x, relative_x, width))
        # print("ymin %f, ymax %f, center_y %f, cell_y %f, relative %f, height %f" % (ymin, ymax, center_y, cell_y, relative_y, height))
        return (relative_x/self.image_size, relative_y/self.image_size, int(cell_x)-1, int(cell_y)-1, width/self.image_size, height/self.image_size)

    def __len__(self):
        return len(self.annotation_path)
    def __getitem__(self, idx):
        label = np.zeros((self.S, self.S, self.B*5+13))

        xml_path = self.annotation_path[idx]
        xml_data = self._xml_parser(xml_path)

        img_path = self.images_path[idx]
        img_size = xml_data["size"]

        image, ratio_x, ratio_y = self._image_processing(img_path, xml_data)

        if "object" not in xml_data:
            pass
        else:
            if type(xml_data["object"]) == list:
                for i, ob in enumerate(xml_data["object"]):
                    object_class = np.zeros(13)
                    x, y, cell_x, cell_y, w, h =self._get_reat_axis(ob["bndbox"], ratio_x, ratio_y)
                    # Each cell has B bounding box
                    for b in range(self.B):
                        label[cell_x, cell_y, b*5:(b+1)*5] = np.array([x, y, w, h, 1])
                    object_name = ob["name"]
                    object_class[classes.index(object_name)] = 1
                    label[cell_x, cell_y, -13:] = object_class
            else:
                ob = xml_data["object"]
                object_class = np.zeros(13)
                x, y, cell_x, cell_y, w, h = self._get_reat_axis(ob["bndbox"], ratio_x, ratio_y)

                for b in range(self.B):
                    label[cell_x, cell_y, b*5:(b+1)*5] = np.array([x, y, w, h, 1])
                object_name = ob["name"]
                object_class[classes.index(object_name)] = 1
                label[cell_x, cell_y, -13:] = object_class

        sample = {"image":np.array(image), "label":label}

        if self.transform:
            sample = self.transform(sample)
        return sample

class ICIPClassifierset(Dataset):
    "The dataset to train the classifier of yolo"
    def __init__(self, csv_file, img_dir, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        row = self.csv_file.iloc[idx, :]

        image = None
        img_name = row[0]
        label = row[1]
        
        img_path = join(self.img_dir, img_name)
        if isfile(img_path):
            try:
                image = PIL.Image.open(img_path)
                image = np.array(image)
            except IOError:
                print("cannot create array for %s" % img_name)
        sample = {'image':image, 'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample
