import torch
import torch.nn as nn

CONV_HIDDEN_CHANNELS = 128
FC_HIDDEN_SIZE = 512

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = x + y;
        y = self.bn2(y)
        y = self.relu(y)
        return y


class NekoFoldNetwork(nn.Module):

    def __init__(self) -> None:
        super(NekoFoldNetwork, self).__init__()

        self.card_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=CONV_HIDDEN_CHANNELS,
                kernel_size=(4,3),
                stride=1,
                padding=1,
            ),
            [ResidualBlock(CONV_HIDDEN_CHANNELS, (4,3)) * 16]
        )

        self.action_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=CONV_HIDDEN_CHANNELS,
                kernel_size=(1,1),
                stride=1,
                padding=0,
            ),
            [ResidualBlock(CONV_HIDDEN_CHANNELS, (4,3)) * 16]
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * CONV_HIDDEN_CHANNELS * 13 * 4, FC_HIDDEN_SIZE),
            nn.BatchNorm1d(FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE),
            nn.BatchNorm1d(FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE),
            nn.BatchNorm1d(FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE),
            nn.BatchNorm1d(FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, FC_HIDDEN_SIZE),
            nn.BatchNorm1d(FC_HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(FC_HIDDEN_SIZE, 6),
            nn.Softmax(dim=1)
        )

    def forward(self,card_features,action_feactures):
        card_features = self.card_conv(card_features)
        action_feactures = self.action_conv(action_feactures)
        x = torch.cat([card_features, action_feactures], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        action = self.action_head(x)
        return action

