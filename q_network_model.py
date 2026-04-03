import torch.nn as nn
import torch


class QNetworkMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetworkMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        q_values = self.layer3(x)
        return q_values


class QNetworkCNN(nn.Module):
    def __init__(self, in_channels, width, height, hidden_size, output_size):
        super(QNetworkCNN, self).__init__()
        # shape: 4*width*height

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1)
        # shape: 32*width*height
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        # shape: 64*width*height
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        # shape: 64*width*height
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        # shape: 64*width*height
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        # shape: 64*width*height
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(64 * width * height, hidden_size)
        self.act6 = nn.ReLU()

        self.fc7 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)

        x = x.flatten(start_dim=1)

        x = self.fc6(x)
        x = self.act6(x)

        q_values = self.fc7(x)
        return q_values
