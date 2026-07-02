import torch
import torch.nn as nn


class ResGroupedMLP1(nn.Module):
    def __init__(self, group_size, feature_dim, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(group_size * feature_dim * output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, input1, input2, input3, input4):
        batch_size, group_size, feature_dim = input1.shape  # batch_size = 10

        input1_flat = input1.view(batch_size, -1)
        input2_flat = input2.view(batch_size, -1)
        input3_flat = input3.view(batch_size, -1)
        input4_flat = input4.view(batch_size, -1)

        x = torch.cat((input1_flat, input2_flat, input3_flat, input4_flat), dim=1)

        # 1st residual block
        res = x
        x = self.fc1(x)
        x = self.activation(x)
        x += res

        # 2nd residual block
        res = x
        x = self.fc2(x)
        x = self.activation(x)
        x += res

        # 3rd residual block
        res = x
        x = self.fc3(x)
        x = self.activation(x)
        x += res

        # 4th residual block
        res = x
        x = self.fc4(x)
        x = self.activation(x)
        x += res

        x = self.fc5(x)
        return x


class DenseGroupedMLP(nn.Module):
    def __init__(self, group_size, feature_dim, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(group_size * feature_dim * output_size, hidden_size)
        self.fc2 = nn.Linear(
            hidden_size + group_size * feature_dim * output_size, hidden_size
        )
        self.fc3 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc4 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc5 = nn.Linear(hidden_size * 5, output_size)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, input1, input2, input3, input4):
        batch_size, group_size, feature_dim = input1.shape

        input1_flat = input1.view(batch_size, -1)
        input2_flat = input2.view(batch_size, -1)
        input3_flat = input3.view(batch_size, -1)
        input4_flat = input4.view(batch_size, -1)

        x = torch.cat((input1_flat, input2_flat, input3_flat, input4_flat), dim=1)
        out1 = self.activation(self.fc1(x))
        out2 = self.activation(self.fc2(torch.cat((x, out1), dim=1)))
        out3 = self.activation(self.fc3(torch.cat((x, out1, out2), dim=1)))
        out4 = self.activation(self.fc4(torch.cat((x, out1, out2, out3), dim=1)))
        out = self.fc5(torch.cat((x, out1, out2, out3, out4), dim=1))
        return out


class SEGroupedMLP(nn.Module):
    def __init__(self, group_size, feature_dim, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(group_size * feature_dim * output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.se = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, input1, input2, input3, input4):
        batch_size, group_size, feature_dim = input1.shape

        input1_flat = input1.view(batch_size, -1)
        input2_flat = input2.view(batch_size, -1)
        input3_flat = input3.view(batch_size, -1)
        input4_flat = input4.view(batch_size, -1)

        x = torch.cat((input1_flat, input2_flat, input3_flat, input4_flat), dim=1)

        # 1st block
        res = x
        x = self.activation(self.fc1(x))
        x = x * self.se(x)  # Apply SE block
        x = x + res

        # 2nd block
        res = x
        x = self.activation(self.fc2(x))
        x = x * self.se(x)
        x = x + res

        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x


class GroupedMLP1(nn.Module):
    def __init__(self, group_size, feature_dim, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(group_size * feature_dim * output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, input1, input2, input3, input4):
        batch_size, group_size, feature_dim = input1.shape  # batch_size = 10
        # print("input1 : ", input1)

        # (batch, group_size, feature_dim) -> (batch, group_size * feature_dim)
        input1_flat = input1.view(batch_size, -1)
        input2_flat = input2.view(batch_size, -1)
        input3_flat = input3.view(batch_size, -1)
        input4_flat = input4.view(batch_size, -1)

        # print("input1_flat : ", input1_flat)
        # print("input1_flat.shape : ", input1_flat.shape)  # torch.Size([10, 15])

        # (batch, group_size * feature_dim * 4)
        x = torch.cat((input1_flat, input2_flat, input3_flat, input4_flat), dim=1)

        # print("GroupedMLP1_cat.shape : ", x.shape)  # torch.Size([10, 60])
        # print("GroupedMLP1_cat : ", x)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x
