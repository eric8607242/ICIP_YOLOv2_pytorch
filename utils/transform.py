import torch
import numpy as np

class ToTensor(object):

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        #label = np.ndarray(label)

        
        image = image.transpose((2, 0, 1))
        return{
                "image":torch.from_numpy(image),
                "label":label
                }

class Normalize(object):
    def __call__(self, sample):
        image, label = sample["image"].astype(float), sample["label"]
        

        image_mean = np.mean(image, axis=(0, 1))
        image_std = np.std(image, axis=(0, 1))

        image = (image-image_mean) / (image_std + 0.00001)

        return{
                "image":image,
                "label":label
                }

