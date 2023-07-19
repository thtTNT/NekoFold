from dataclasses import asdict
from enum import Enum

import torch
from torch import nn
from pymongo import MongoClient
import json

from entity import GameState, Action, CARDS, GameRound
import numpy as np
from train import Network, encode_state


def convert_object_to_dict(obj):
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.append(convert_object_to_dict(item))
        return result
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = convert_object_to_dict(value)
        return obj
    elif issubclass(obj.__class__, Enum):
        return obj.name
    elif hasattr(obj, "__dict__"):
        return convert_object_to_dict(obj.__dict__)
    else:
        return obj


class Player:
    def take_action(self, id: int, state: GameState, legal_actions: list[Action]) -> Action:
        if Action.CHECK in legal_actions:
            return Action.CHECK
        elif Action.CALL in legal_actions:
            return Action.CALL
        elif Action.ALL_IN in legal_actions:
            return Action.ALL_IN
        else:
            return Action.FOLD

    def is_training_mode(self) -> bool:
        return False

    def train(self, records):
        pass


class AIPlayer(Player):
    def __init__(self, training_mode: bool):
        self.training_mode = training_mode
        self.network = Network()
        self.network.load_state_dict(torch.load("model_gen001.pt"))
        self.mongo_client = MongoClient("mongodb://192.168.0.150:27017/")

    def take_action(self, player_id: int, state: GameState, legal_actions: list[Action]) -> Action:
        data = encode_state(state)
        props = self.network(torch.from_numpy(data)).detach().numpy()
        if Action.FOLD not in legal_actions:
            props[0] = 0
        if Action.CHECK not in legal_actions:
            props[1] = 0
        if Action.CALL not in legal_actions:
            props[2] = 0
        if Action.RAISE_HALF_POT not in legal_actions:
            props[3] = 0
        if Action.RAISE_POT not in legal_actions:
            props[4] = 0
        if Action.ALL_IN not in legal_actions:
            props[5] = 0
        props = props / np.sum(props)
        return np.random.choice(
            [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN], p=props)

    def is_training_mode(self) -> bool:
        return self.training_mode

    def train(self, records):
        if not self.training_mode:
            return
        for record in records:
            self.mongo_client["beta-fold"].record.insert_one({
                "state": record["state"].to_dict(),
                "actions": record["actions"]
            })

    def save(self, it: int):
        if it % 128 == 0:
            torch.save(self.network.state_dict(), "network_gen1.pth")
