import torch
import torch.nn as nn


class GroupedMLP1(nn.Module):
    def __init__(self, hidden_dim, hidden_size, output_size):
        super().__init__()
        # 인코더 4개 concat → hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, input1, input2, input3, input4):
        # input1~4: (batch, hidden_dim)
        x = torch.cat((input1, input2, input3, input4), dim=1)  # (batch, hidden_dim*4)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x
