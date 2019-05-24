from os import listdir
from os.path import isfile, join

import numpy as np
import xmltodict
import PIL.Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from utils.transform import ToTensor
from utils.util import parse_xml

class Detectionset(Dataset):
    def __init__(self,
                kmean,
                classes,
                train_image_folder, 
                train_annot_folder, 
                input_size, 
                S=7, 
                B=2, 
            ):
        self.train_image_folder = train_image_folder
        self.train_annot_folder = train_annot_folder
        self.input_size = input_size
        self.transform = transforms.Compose([ToTensor()])

        self.classes = classes
        
        self.S = S
        self.B = B

        self.images_path = [join(self.train_image_folder, f) for f in listdir(self.train_image_folder) if isfile(join(self.train_image_folder, f))]
        self.annotation_path = [join(self.train_annot_folder, f) for f in listdir(self.train_annot_folder) if isfile(join(self.train_annot_folder, f))]

        self.images_path.sort()
        self.annotation_path.sort()

        self.kmeans = kmean
                        
    def _image_processing(self, img_path, xml_data):
        width = xml_data["size"]["width"]
        height = xml_data["size"]["height"]

        image = PIL.Image.open(img_path)
        image = image.convert('RGB')
        image = image.resize((self.input_size, self.input_size), PIL.Image.ANTIALIAS)

        ratio_x = self.input_size / int(width)
        ratio_y = self.input_size/ int(height)

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

        cell_size = self.input_size / self.S
        cell_x = center_x // cell_size
        cell_y = center_y // cell_size
        
        relative_x = (center_x - cell_x*cell_size)/cell_size
        relative_y = (center_y - cell_y*cell_size)/cell_size
        
        return (relative_x, 
                relative_y, 
                int(cell_x)-1, 
                int(cell_y)-1, 
                width/self.input_size, 
                height/self.input_size)

    def __len__(self):
        return len(self.annotation_path)

    def __getitem__(self, idx):
        label = np.zeros((self.S, self.S, self.B*5+13))

        xml_path = self.annotation_path[idx]
        img_path = self.images_path[idx]

        xml_data = parse_xml(xml_path)
        img_size = xml_data["size"]

        image, ratio_x, ratio_y = self._image_processing(img_path, xml_data)

        for i, ob in enumerate(xml_data["object"]):
            object_class = np.zeros(13)
            x, y, cell_x, cell_y, w, h =self._get_reat_axis(ob["bndbox"], ratio_x, ratio_y)

            # Each cell has B bounding box
            b_id = self.kmeans.predict([[w, h]])[0]
            label[cell_x, cell_y, b_id*5:(b_id+1)*5] = np.array([x, y, w, h, 1])

            object_name = ob["name"]
            object_class[self.classes.index(object_name)] = 1

            label[cell_x, cell_y, -13:] = object_class

        sample = {"image":np.array(image), "label":label}

        if self.transform:
            sample = self.transform(sample)
            
        return sample

