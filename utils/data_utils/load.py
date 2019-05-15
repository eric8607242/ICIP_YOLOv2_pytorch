from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

import xml.etree.ElementTree as ET

import cv2
import csv

import PIL.Image

train_classifier_dir = "./data/train_classifier/"
label_path = "./data/train_list.csv"
classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower']


def load_data(start, N=7935, H=224, W=224, C=3):
    train_data = np.zeros((N, H, W, C))
    label_data = np.zeros((N, 1))

    with open(label_path, newline='') as csvfile:
        rows = list(csv.reader(csvfile))
        rows = rows[start:start+N]

        for i, row in enumerate(rows):
            filename = row[0]
            label = row[1]

            filepath = join(train_classifier_dir, filename)
            if isfile(filepath):
                try:
                    img = PIL.Image.open(filepath)
                    train_data[i] = np.array(img)
                except IOError:
                    print("cannot create array for %s" % filename)
            label_data[i] = label
    return train_data, label_data



