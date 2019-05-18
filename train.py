from utils.data_utils.dataset import ICIPClassifierset, ICIPDetectionset
from utils.resnet import resnet18, Flatten
from utils.lossfn import LossFunction
from utils.show_img import show_img

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.data_utils.transform import ToTensor, Normalize

class DetectionModel:
    def __init__(self):
        self.net = resnet18()

    def _init_model(self):

        resnet = models.resnet18(pretrained=True)
        resnet_dict = resnet.state_dict()
        net_dict = self.net.state_dict()

        for k in resnet_dict.keys():
            if k in net_dict.keys():
                net_dict[k] = resnet_dict[k]
        self.net.load_state_dict(net_dict)

    def train(self):
        pass

class ClassifierModel:
    def __init__(self):
        self.net = resnet18()
        self._init_model()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-3)

        self.transform = transforms.Compose([ToTensor()])
        self.data = ICIPClassifierset("./data/train_list.csv", "./data/train_classifier/", transform=self.transform)
        self.dataloader = DataLoader(self.data, batch_size=100, shuffle=True, num_workers=4)

    def _init_model(self):

        resnet = models.resnet18(pretrained=True)
        resnet_dict = resnet.state_dict()
        net_dict = self.net.state_dict()

        for k in resnet_dict.keys():
            if k in net_dict.keys():
                net_dict[k] = resnet_dict[k]
        self.net.load_state_dict(net_dict)
        self.net = nn.Sequential(
                    self.net,
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(1127, 13)
                )

    def train(self, num_epochs=30):       
        print("training")
        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.9
            for i, local_batch in enumerate(self.dataloader):
                batch_data = local_batch["image"].float()
                batch_label = local_batch["label"]

                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                
                # Forward pass
                outputs = self.net(batch_data)

                loss = self.loss(outputs, batch_label)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                self.optimizer.step()
                
                if (i+1) % 5 == 0:
                    print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs,i+1, loss.item()))
        torch.save(self.net.state_dict(),'yolo_classification.pth')

