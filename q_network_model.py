import torch.nn as nn


class QNetworkCNN(nn.Module):
    def __init__(self, in_channels, hidden_size, output_size):
        super(QNetworkCNN, self).__init__()
        # shape: 4*width*height

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1)
        # shape: 32*width*height
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        # shape: batch_size*64*width*height
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        # shape: batch_size*128*width*height
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        # shape: batch_size*128*width*height
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        # shape: batch_size*128*width*height
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(128, hidden_size)
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

        #x= x.flatten(start_dim=1)
        x = x.mean(dim=[-2,-1])

        x = self.fc6(x)
        x = self.act6(x)

        q_values = self.fc7(x)
        return q_values
