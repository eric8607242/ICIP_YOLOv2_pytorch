import torch
import torch.nn as nn
import torch.nn.functional as F

class Siannet(nn.Module):
    def __init__(self):
        super(Siannet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 72, 4, 2)
        self.conv4 = nn.Conv2d(72, 96, 3)
        self.conv5 = nn.Conv2d(96, 108, 3)
        self.fc1 = nn.Linear(23*23*108, 5000)
        self.fc2 = nn.Linear(5000, 1200)
        self.fc3 = nn.Linear(1200, 300)
        self.fc4 = nn.Linear(300, 13)

    def forward(self, x):
        x = self.bn1(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s
        return num_features

