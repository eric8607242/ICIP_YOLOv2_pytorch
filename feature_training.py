from utils.data_utils.dataset import ICIPClassifierset, ICIPDetectionset
from utils.model import Siannet
from utils.vgg_model import vgg11, vgg11_bn
from utils.lossfn import LossFunction
from utils.show_img import show_img

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.data_utils.transform import ToTensor, Normalize



#net = Siannet()
#net.cuda()
print("net create")
net = models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 13)
            #nn.LogSigmoid()
        )
net_d = models.resnet18(pretrained=True)
for i, param in enumerate(net_d.parameters()):
    if i < 40:
        param.requires_grad = False

num_ftrs_d = net_d.fc.in_features
net_d.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 7*7*23),
            nn.Sigmoid()
        )


net.load_state_dict(torch.load("./yolo.pth"))
state_dict = net.state_dict()
state_dict_d = net_d.state_dict()

for k in state_dict.keys():
    if k in state_dict_d.keys() and not k.startswith('fc.4'):
        state_dict_d[k] = state_dict[k]

net_d.load_state_dict(state_dict_d)
#new_state_dict = vgg.state_dict()

#net = vgg11_bn()
#dd = net.state_dict()
#for k in new_state_dict.keys():
#    print(k)
#    if k in dd.keys() and k.startswith('features'):
#        dd[k] = new_state_dict[k]
#net.load_state_dict(dd)

net_d.cuda()
net_d = net_d.float()

print("data prepare")

transform = transforms.Compose(
    [
        #Normalize(),
        ToTensor()
     ])

criterion = nn.CrossEntropyLoss()
criterion1 = LossFunction(7, 2, 13, 5, 0.5)
optimizer = torch.optim.Adam(net_d.parameters(), lr=5e-3)
data = ICIPClassifierset("./data/train_list.csv", "./data/train_classifier/", transform=transform)
data1 = ICIPDetectionset("./data/train_cdc/train_images/", "./data/train_cdc/train_annotations/", 448, transform=transform)
dataloader = DataLoader(data1, batch_size=100, shuffle=True, num_workers=4)

print("training")
num_epochs = 100 
for epoch in range(num_epochs):
    if epoch % 5 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
    for i, local_batch in enumerate(dataloader):
        batch_data = local_batch["image"].float()
        batch_label = local_batch["label"]

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        
        # Forward pass
        outputs = net_d(batch_data)
        outputs = outputs.view(-1, 7, 7, 23)

        loss = criterion1(outputs, batch_label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in net_d.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        optimizer.step()
        
        if (i+1) % 5 == 0:
            show_img(batch_data[0].cpu(), outputs[0].cpu(), batch_label[0].cpu())
            l = batch_label[0, :, :, :10]
            o = outputs[0, :, :, :10]
            l = l.contiguous().view(-1, 5)
            o = o.contiguous().view(-1, 5)

            print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'.format(epoch+1, num_epochs,i+1, loss.item()))

torch.save(net_d.state_dict(),'yolo_d.pth')
