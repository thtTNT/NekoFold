import math
import sys

import numpy as np
import torch
import torch.nn as nn
from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader

from entity import GameState, GameRound, CARDS

EPOCH = 32
IT_PER_EPOCH = 64
BATCH_SIZE = 32

mongo_client = MongoClient("mongodb://192.168.0.150:27017/")
db = mongo_client["beta-fold"]


def convert_card_to_matrix(card):
    matrix = np.zeros(52)
    matrix[CARDS.index(card)] = 1
    return matrix


def encode_state(state: GameState):
    player_id = state.index
    data = np.array([])
    if state.round == GameRound.BEFORE_FOLD:
        data = np.append(data, [1, 0, 0, 0])
    elif state.round == GameRound.AFTER_FOLD:
        data = np.append(data, [0, 1, 0, 0])
    elif state.round == GameRound.TURN:
        data = np.append(data, [0, 0, 1, 0])
    elif state.round == GameRound.RIVER:
        data = np.append(data, [0, 0, 0, 1])
    data = np.append(data, convert_card_to_matrix(state.private_cards[player_id][0]))
    data = np.append(data, convert_card_to_matrix(state.private_cards[player_id][1]))
    for i in range(len(state.community_cards)):
        data = np.append(data, convert_card_to_matrix(state.community_cards[i]))
    data = np.append(data, ((5 - len(state.community_cards)) * 52) * [0])
    data = np.append(data, state.pot / 2000)
    data = np.append(data, state.chips[player_id] / 2000)
    data = np.append(data, state.chips_needed_to_call / 2000)
    data = np.append(data, state.chips_already_call[player_id] / 2000)
    data = np.append(data, state.chips_needed_to_call / state.pot)
    data = np.append(data, state.chips_already_call[player_id] / state.pot)
    data = np.append(data, state.chips[player_id] / state.pot)
    data = np.append(data, [0] * (512 - data.shape[0]))
    data = np.cast['float32'](data)
    return data


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_size = 1024

        self.fc1 = nn.Linear(512, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc6 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc7 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc8 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc9 = nn.Linear(self.hidden_size, 6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc9(x)
        x = self.softmax(x)
        return x


def encode_actions(actions, player_index: int):
    encoded_actions = [math.nan] * 6
    if "FOLD" in actions:
        encoded_actions[0] = actions["FOLD"][player_index]
    if "CHECK" in actions:
        encoded_actions[1] = actions["CHECK"][player_index]
    if "CALL" in actions:
        encoded_actions[2] = actions["CALL"][player_index]
    if "RAISE_HALF_POT" in actions:
        encoded_actions[3] = actions["RAISE_HALF_POT"][player_index]
    if "RAISE_POT" in actions:
        encoded_actions[4] = actions["RAISE_POT"][player_index]
    if "ALL_IN" in actions:
        encoded_actions[5] = actions["ALL_IN"][player_index]
    return encoded_actions


class MongodbDataset(Dataset):

    def __init__(self):
        self.document_ids = list(db.record.find({}, "_id"))

    def __len__(self):
        return len(self.document_ids)

    def __getitem__(self, index):
        data = db.record.find_one(self.document_ids[index])
        encoded_state = encode_state(GameState.from_dict(data["state"]))
        encoded_actions = encode_actions(data["actions"], data["state"]["index"])
        return torch.tensor(encoded_state), torch.tensor(encoded_actions)


def train():
    dataset = MongodbDataset()
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last=True)
    network = Network()
    loss_function = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    for epoch in range(EPOCH):
        print("epoch: ", epoch)
        it = 0
        total_loss = 0
        for data in dataloader:
            state, action_value = data
            y_pred = network(state)
            action_props = torch.clone(y_pred).detach()
            action_props = torch.where(torch.isnan(action_value), torch.zeros_like(action_props), action_props)
            action_props = action_props / action_props.sum(dim=1).reshape(BATCH_SIZE, 1)
            virtual_expected_value = (action_value * action_props).nansum(dim=1).reshape(BATCH_SIZE, 1)
            regret = action_value - virtual_expected_value
            regret = torch.where(regret < 0, torch.zeros_like(regret), regret)
            regret = torch.where(torch.isnan(regret), torch.zeros_like(regret), regret)
            # convert regret to probability
            y_true = regret / (regret.sum(dim=1).reshape(BATCH_SIZE, 1))
            loss = loss_function(torch.log(y_pred), y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(it, "..", end="")
            it = it + 1
            total_loss = total_loss + loss.item()
        print("loss: ", total_loss / it)
        torch.save(network.state_dict(), "model_gen001.pt")

if __name__ == '__main__':
    train()
