import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        hidden1_size = input_size // 20
        hidden2_size = hidden1_size // 2
        hidden3_size = hidden2_size // 2

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x