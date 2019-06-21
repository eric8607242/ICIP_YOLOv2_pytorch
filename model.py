from utils.resnet import resnet18
from lossfn import LossFunction
from utils.show_img import show_img

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler


class DetectionModel:
    def __init__(self, 
                dataloader,
                anchor_box,
                train_image_folder,
                train_annot_folder,
                pretrained_weights,
                save_weight_name, 
                batch_size,
                object_scale,
                no_object_scale,
                coord_scale,
                class_scale
            ):
        self.pretrained_weights = pretrained_weights
        self.save_weight_name = save_weight_name

        self.net = resnet18()
        self._init_model()

        self.loss = LossFunction(14, 5, 13, object_scale, no_object_scale, coord_scale, class_scale)
        self.dataloader = dataloader

        self.anchor_box = anchor_box

    def _init_model(self):
        if self.pretrained_weights is "" :
            class_resnet = models.resnet18(pretrained=True)
            class_dict = class_resnet.state_dict()

            net_dict = self.net.state_dict()
            class_dict = {k: v for k, v in class_dict.items() if k in net_dict}

            net_dict.update(class_dict)
            self.net.load_state_dict(net_dict)
        else:
            self.net.load_state_dict(torch.load(self.pretrained_weights))

        for net_name, net_param in self.net.named_parameters():
            if net_name.startswith("layer1") or net_name.startswith("layer2"):
                net_param.requires_grad=False
            else:
                net_param.requires_grad=True
        self.net.cuda()

    def train(  self, 
                epochs=50, 
                learning_rate=1e-3,
                step_size=5,
                decay_ratio=0.9,
            ):

        print("-------- Training Step Start --------")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_ratio)

        for epoch in range(epochs):
            for i, local_batch in enumerate(self.dataloader):
                batch_data = local_batch["image"].float()
                batch_label = local_batch["label"]

                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                
                # Forward pass
                outputs = self.net(batch_data)

                loss = self.loss(outputs, batch_label, self.anchor_box)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                optimizer.step()
                scheduler.step()
                
                if (i+1) % 5 == 0:
                    show_img(batch_data[0].cpu(), outputs[0].cpu(), batch_label[0].cpu(), self.anchor_box, 14, 5)
                    print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, epochs,i+1, loss.item()))
        torch.save(self.net.state_dict(),self.save_weight_name)
