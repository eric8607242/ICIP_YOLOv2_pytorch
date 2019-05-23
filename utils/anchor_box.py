from os import listdir
from os.path import isfile, join

from sklearn.cluster import KMeans

import numpy as np
import xmltodict
import pickle

from utils.util import parse_xml


def calculate_anchor(input_size, train_annot_folder, saved_kmean_name):
    annot_path_list = [join(train_annot_folder, f) for f in listdir(train_annot_folder) if isfile(join(train_annot_folder, f))]
    annot_path_list.sort()

    kmean_list = []
    for annot_name in annot_path_list:
        xml_data = parse_xml(annot_name)

        ratio_x, ratio_y = get_ratio(input_size, xml_data)
        for ob in xml_data["object"]:
            w, h = get_wh(input_size, ob["bndbox"], ratio_x, ratio_y)
            kmean_list.append([w, h])

    kmean_list = np.asarray(kmean_list)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(kmean_list)
    pickle.dump(kmeans, open(saved_kmean_name, 'wb'))


def get_kmean(pretrained_kmean):
    kmeans = pickle.load(open(pretrained_kmean, 'rb'))
    return kmeans


def get_anchor(kmean):
    anchor_box = kmean.cluster_centers_
    return anchor_box


def get_ratio(input_size, xml_data):
    width = xml_data["size"]["width"]
    height = xml_data["size"]["width"]

    ratio_x = input_size / int(width)
    ratio_y = input_size / int(height)

    return ratio_x, ratio_y


def get_wh(input_size, bndbox, ratio_x, ratio_y):
    xmin = int(bndbox["xmin"]) * ratio_x
    xmax = int(bndbox["xmax"]) * ratio_x

    ymin = int(bndbox["ymin"]) * ratio_y
    ymax = int(bndbox["ymax"]) * ratio_y

    width = (xmax-xmin) / input_size
    height = (ymax-ymin) / input_size

    return width, height

    
