import torch.nn.functional as F
from torch import nn, relu

class Model(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, output_size):
        super(Model, self).__init__()
        self.entry = nn.Linear(input_size, hidden_1)
        self.hidden = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, output_size)

    def forward(self, x):
        out = relu(self.entry(x))
        out = relu(self.hidden(out))
        out = self.out(out)
        return out


class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(32*24*24, 256)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 32*24*24)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
