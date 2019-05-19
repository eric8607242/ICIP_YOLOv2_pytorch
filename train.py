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
        self._init_model()
        self.sigmoid = nn.Sequential(
                    nn.Softmax()
                )

        self.loss = LossFunction(14, 2, 13, 5, 0.5)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=5e-3, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-4)

        self.transform = transforms.Compose([ToTensor()])
        self.data = ICIPDetectionset("./data/train_cdc/train_images/", "./data/train_cdc/train_annotations/", 448, 14, transform=self.transform)
        self.dataloader = DataLoader(self.data, batch_size=32, shuffle=True, num_workers=4)

    def _init_model(self):

        #class_resnet = resnet18()
        #class_resnet.load_state_dict(torch.load('yolo_classification.pth'))
        class_resnet = models.resnet18(pretrained=True)
        class_dict = class_resnet.state_dict()

        net_dict = self.net.state_dict()
        class_dict = {k: v for k, v in class_dict.items() if k in net_dict}
        net_dict.update(class_dict)
        self.net.load_state_dict(net_dict)
        for net_name, net_param in self.net.named_parameters():
            if net_name.startswith("layer4") or net_name.startswith("layer3") or net_name.startswith("layer2") or net_name.startswith("layer1"):
                net_param.requires_grad=False
            else:
                net_param.requires_grad=True
        self.net.cuda()


    def train(self, num_epochs=50):
        print("training")
        for epoch in range(num_epochs):
            if epoch % 5 == 0 and epoch <= 50:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
            for i, local_batch in enumerate(self.dataloader):
                batch_data = local_batch["image"].float()
                batch_label = local_batch["label"]

                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                
                # Forward pass
                outputs = self.net(batch_data)
                #outputs = self.sigmoid(outputs)

                loss = self.loss(outputs, batch_label)
                for param in self.net.parameters():
                    if param.grad is not None:
                        pass
                        #param.grad.data.clamp_(-1, 1)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                self.optimizer.step()
                
                if (i+1) % 5 == 0:
                    show_img(batch_data[0].cpu(), outputs[0].cpu(), batch_label[0].cpu(), 14)
                    print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs,i+1, loss.item()))
        torch.save(self.net.state_dict(),'yolo_detection.pth')

class ClassifierModel:
    def __init__(self):
        self.net = resnet18()
        self.classifier = nn.Sequential(
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(4508, 13)
                )
        self.classifier.cuda()
        self._init_model()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-3)

        self.transform = transforms.Compose([ToTensor()])
        self.data = ICIPClassifierset("./data/train_list.csv", "./data/train_classifier/", transform=self.transform)
        self.dataloader = DataLoader(self.data, batch_size=50, shuffle=True, num_workers=4)

    def _init_model(self):

        resnet = models.resnet18(pretrained=True)
        resnet_dict = resnet.state_dict()
        net_dict = self.net.state_dict()

        resnet_dict = {k: v for k, v in resnet_dict.items() if k in net_dict}
        net_dict.update(resnet_dict)
        self.net.load_state_dict(net_dict)

        for net_name, net_param in self.net.named_parameters():
            if net_name.startswith("layer4") or net_name.startswith("layer3") or net_name.startswith("layer2") or net_name.startswith("layer1"):
                net_param.requires_grad=False
            else:
                net_param.requires_grad=True
        self.net.cuda()

    def train(self, num_epochs=5):       
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
                outputs = self.classifier(outputs)

                loss = self.loss(outputs, batch_label)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                self.optimizer.step()
                
                if (i+1) % 5 == 0:
                    print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs,i+1, loss.item()))
        torch.save(self.net.state_dict(),'yolo_classification.pth')
if __name__=="__main__":
    net = DetectionModel()
    net.train(num_epochs=100)
