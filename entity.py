from dataclasses import dataclass
from enum import Enum

CARDS = ["SA", "SK", "SQ", "SJ", "ST", "S9", "S8", "S7", "S6", "S5", "S4", "S3", "S2",
         "HA", "HK", "HQ", "HJ", "HT", "H9", "H8", "H7", "H6", "H5", "H4", "H3", "H2",
         "DA", "DK", "DQ", "DJ", "DT", "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2",
         "CA", "CK", "CQ", "CJ", "CT", "C9", "C8", "C7", "C6", "C5", "C4", "C3", "C2"]


class GameRound(Enum):
    BEFORE_FOLD = 0
    AFTER_FOLD = 1
    TURN = 2
    RIVER = 3


class Action(Enum):
    FOLD = 0
    CALL = 1
    CHECK = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    ALL_IN = 5


class GameState:
    round: GameRound = None
    community_cards: list[str] = None
    private_cards: list[list[str]] = None
    chips: list[int] = None
    chips_needed_to_call: int = None
    chips_already_call: list[int] = None
    left_cards: list[str] = None
    dealer: int = None
    fold: list[bool] = None
    allin: list[bool] = None
    max_win: list[int] = None
    index: int = None
    minimum_bet: int = None
    # 所有玩家累积下注
    total_bet: list[int] = None
    pot: int = 0
    # 用于判断是否每个玩家都已经操作
    ready_next_round: bool = False

    def to_dict(self):
        return {
            "round": self.round.name,
            "community_cards": self.community_cards,
            "private_cards": self.private_cards,
            "chips": self.chips,
            "chips_needed_to_call": self.chips_needed_to_call,
            "chips_already_call": self.chips_already_call,
            "left_cards": self.left_cards,
            "dealer": self.dealer,
            "fold": self.fold,
            "allin": self.allin,
            "max_win": self.max_win,
            "index": self.index,
            "minimum_bet": self.minimum_bet,
            "total_bet": self.total_bet,
            "pot": self.pot,
            "ready_next_round": self.ready_next_round
        }

    @staticmethod
    def from_dict(source):
        state = GameState()
        state.round = GameRound.__members__[source["round"]]
        state.community_cards = source["community_cards"]
        state.private_cards = source["private_cards"]
        state.chips = source["chips"]
        state.chips_needed_to_call = source["chips_needed_to_call"]
        state.chips_already_call = source["chips_already_call"]
        state.left_cards = source["left_cards"]
        state.dealer = source["dealer"]
        state.fold = source["fold"]
        state.allin = source["allin"]
        state.max_win = source["max_win"]
        state.index = source["index"]
        state.minimum_bet = source["minimum_bet"]
        state.total_bet = source["total_bet"]
        state.pot = source["pot"]
        state.ready_next_round = source["ready_next_round"]
        return state
