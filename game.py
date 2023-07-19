import copy
import random
import functools

from entity import GameState, Action, GameRound, CARDS
from player import Player
from rule import findPattern, compare_pattern


def array_add(array1, array2):
    if len(array1) != len(array2):
        raise Exception("Lengths of two arrays are not equal.")
    return [array1[i] + array2[i] for i in range(len(array1))]


def array_sub(array1, array2):
    if len(array1) != len(array2):
        raise Exception("Lengths of two arrays are not equal.")
    return [array1[i] - array2[i] for i in range(len(array1))]


def array_div(array, num):
    return [array[i] / num for i in range(len(array))]


def sort_pattern(patterns):
    bucket_patterns = []
    for pattern in patterns:
        flag = False
        for bucket_pattern in bucket_patterns:
            if compare_pattern(pattern[1], bucket_pattern[0][1]) == 0:
                bucket_pattern.append(pattern)
                flag = True
                break
            if compare_pattern(pattern[1], bucket_pattern[0][1]) > 0:
                bucket_patterns.insert(bucket_patterns.index(bucket_pattern), [pattern])
                flag = True
                break
        if not flag:
            bucket_patterns.append([pattern])
    return bucket_patterns


def get_legal_action(player_index: int, state: GameState):
    legal_actions = []
    if state.fold[player_index]:
        return legal_actions
    legal_actions.append(Action.FOLD)
    if state.chips_needed_to_call != state.chips_already_call[player_index] and \
            state.chips[player_index] > state.chips_needed_to_call - state.chips_already_call[player_index]:
        legal_actions.append(Action.CALL)
    if state.chips_needed_to_call == state.chips_already_call[player_index]:
        legal_actions.append(Action.CHECK)
    if int(state.pot / 2) > state.minimum_bet and \
            state.chips[player_index] > state.pot / 2 - state.chips_already_call[player_index]:
        legal_actions.append(Action.RAISE_HALF_POT)
    if state.pot > state.minimum_bet and \
            state.chips[player_index] > state.pot - state.chips_already_call[player_index]:
        legal_actions.append(Action.RAISE_POT)
    if state.chips[player_index] > 0:
        legal_actions.append(Action.ALL_IN)
    return legal_actions


def check_can_enter_next_round(state: GameState):
    if not state.ready_next_round:
        return False
    for i in range(len(state.chips_already_call)):
        if state.chips_already_call[i] != state.chips_needed_to_call and not state.fold[i] and not state.allin[i]:
            return False
    return True


def draw_card(state: GameState) -> str:
    card = random.choice(state.left_cards)
    state.left_cards.remove(card)
    return card


def process_action(state: GameState, player_index: int, action: Action):
    if action == Action.FOLD:
        state.fold[player_index] = True
    elif action == Action.CALL:
        bet_amount = state.chips_needed_to_call - state.chips_already_call[player_index]
        state.total_bet[player_index] += bet_amount
        state.chips[player_index] -= bet_amount
        state.pot += bet_amount
        state.chips_already_call[player_index] = state.chips_needed_to_call
    elif action == Action.CHECK:
        if state.chips_needed_to_call != state.chips_already_call[player_index]:
            raise Exception("Invalid action")
    elif action == Action.RAISE_HALF_POT:
        if int(state.pot / 2) < state.minimum_bet or int(state.pot / 2) <= state.chips_needed_to_call:
            raise Exception("Invalid action")
        bet_amount = int(state.pot / 2) - state.chips_already_call[player_index]
        state.total_bet[player_index] += bet_amount
        state.chips[player_index] -= bet_amount
        state.chips_already_call[player_index] += bet_amount
        state.pot += bet_amount
        state.chips_needed_to_call = state.chips_already_call[player_index]
        state.minimum_bet = bet_amount + state.chips_already_call[player_index]
    elif action == Action.RAISE_POT:
        if state.pot < state.minimum_bet or state.pot <= state.chips_needed_to_call:
            raise Exception("Invalid action")
        bet_amount = state.pot - state.chips_already_call[player_index]
        state.total_bet[player_index] += bet_amount
        state.chips[player_index] -= bet_amount
        state.chips_already_call[player_index] += bet_amount
        state.pot += bet_amount
        state.chips_needed_to_call = state.chips_already_call[player_index]
        state.minimum_bet = bet_amount + state.chips_already_call[player_index]
    elif action == Action.ALL_IN:
        bet_amount = state.chips[player_index]
        state.total_bet[player_index] += bet_amount
        state.chips[player_index] = 0
        state.chips_already_call[player_index] += bet_amount
        state.pot += bet_amount
        state.chips_needed_to_call = max(state.chips_already_call[player_index],
                                         state.chips_needed_to_call)
        state.minimum_bet = max(bet_amount + state.chips_already_call[player_index], state.minimum_bet)
        state.allin[player_index] = True


def calculate_max_win(state: GameState):
    for i in range(len(state.allin)):
        if (not state.allin[i]) or state.chips_already_call[i] == 0:
            continue
        max_win = state.pot
        for j in range(len(state.chips_already_call)):
            if state.chips_already_call[i] < state.chips_already_call[j]:
                max_win -= state.chips_already_call[j] - state.chips_already_call[i]
        state.max_win[i] = max_win


def settle(state: GameState):
    result = [0] * len(state.chips)
    patterns = []
    for i in range(len(state.chips)):
        if state.fold[i]:
            continue
        patterns.append((i, findPattern(state.private_cards[i] + state.community_cards)))

    bucket_patterns = sort_pattern(patterns)

    already_take_pot = 0
    for i in range(len(state.max_win)):
        if state.max_win[i] > state.pot:
            state.max_win[i] = state.pot
    for bucket_pattern in bucket_patterns:
        if already_take_pot == state.pot:
            break
        winner_indexes = [pattern[0] for pattern in bucket_pattern]
        winner_indexes = [winner_index for winner_index in winner_indexes if state.max_win[winner_index] > 0]
        winner_indexes.sort(key=lambda x: state.max_win[x])
        while len(winner_indexes) > 0:
            winner_index = winner_indexes[0]
            win_chip = int(state.max_win[winner_index] / len(winner_indexes))
            max_win = state.max_win[winner_index]
            for i in range(len(winner_indexes)):
                state.chips[winner_indexes[i]] += win_chip
                result[winner_indexes[i]] += win_chip
            for i in range(len(state.max_win)):
                state.max_win[i] -= max_win
            already_take_pot += max_win
            winner_indexes = [winner_index for winner_index in winner_indexes if state.max_win[winner_index] > 0]
    return result


class Game:

    def __init__(self, players: list[Player], initial_chips: list[int]):
        self.records = None
        self.players = players
        self.initial_chips = initial_chips

    def init_state(self):
        state = GameState()
        state.round = GameRound.BEFORE_FOLD
        state.community_cards = []
        state.private_cards = []
        state.pot = 0
        state.chips_needed_to_call = 0
        state.chips_already_call = [0] * len(self.players)
        state.left_cards = CARDS.copy()
        state.dealer = random.randint(0, len(self.players) - 1)
        state.fold = [False] * len(self.players)
        state.allin = [False] * len(self.players)
        state.max_win = [2 ** 32] * len(self.players)
        state.chips = self.initial_chips.copy()
        state.total_bet = [0] * len(self.players)
        random.shuffle(state.left_cards)
        self.deal_cards(state)

        state.pot = 30
        state.chips_needed_to_call = 20
        state.chips_already_call[(state.dealer + 1) % len(self.players)] = 10
        state.chips_already_call[(state.dealer + 2) % len(self.players)] = 20
        state.chips[(state.dealer + 1) % len(self.players)] -= 10
        state.chips[(state.dealer + 2) % len(self.players)] -= 20
        state.minimum_bet = 40
        state.index = (state.dealer + 3) % len(self.players)
        return state

    def play(self):
        self.records = []
        state = self.init_state()
        game_result = self.update_state(state)
        final_result = []
        for record in self.records:
            flag = False
            for result in final_result:
                if result["state"] == record[0]:
                    result["actions"][record[1].name] = record[3]
                    flag = True
                    break
            if not flag:
                final_result.append({"state": record[0], "actions": {}})
                final_result[-1]["actions"][record[1].name] = record[3]
        return game_result, final_result

    def deal_cards(self, state: GameState):
        for i in range(len(self.players)):
            state.private_cards.append([])
            for j in range(2):
                card = draw_card(state)
                state.private_cards[i].append(card)

    def update_state(self, state):
        next_states = []
        expected_value = [0] * len(self.players)
        training_mode = False
        player = self.players[state.index]
        if state.allin[state.index] or state.fold[state.index]:
            next_states.append(state)
        else:
            if player.is_training_mode():
                training_mode = True
                for action in get_legal_action(state.index, state):
                    new_state = copy.deepcopy(state)
                    self.records.append([state, action, new_state, None])
                    process_action(new_state, state.index, action)
                    next_states.append(new_state)
            else:
                action = player.take_action(state.index, state, get_legal_action(state.index, state))
                process_action(state, state.index, action)
                next_states.append(state)
        if state.round == GameRound.BEFORE_FOLD:
            for next_state in next_states:
                if next_state.index == (next_state.dealer + 2) % len(self.players):
                    next_state.ready_next_round = True
                if check_can_enter_next_round(next_state):
                    calculate_max_win(next_state)
                    next_state.index = (next_state.dealer + 1) % len(self.players)
                    next_state.community_cards.append(draw_card(next_state))
                    next_state.community_cards.append(draw_card(next_state))
                    next_state.community_cards.append(draw_card(next_state))
                    next_state.chips_already_call = [0] * len(self.players)
                    next_state.chips_needed_to_call = 0
                    next_state.minimum_bet = 20
                    next_state.ready_next_round = False
                    next_state.round = GameRound.AFTER_FOLD
                else:
                    next_state.index = (next_state.index + 1) % len(self.players)
                result = self.update_state(next_state)
                if training_mode:
                    for record in self.records:
                        if record[0] == state and record[2] == next_state:
                            record[3] = result
                expected_value = array_add(expected_value, result)
            return array_div(expected_value, len(next_states))
        elif state.round == GameRound.AFTER_FOLD:
            for next_state in next_states:
                if next_state.index == next_state.dealer:
                    next_state.ready_next_round = True
                if check_can_enter_next_round(next_state):
                    calculate_max_win(next_state)
                    next_state.index = (next_state.dealer + 1) % len(self.players)
                    next_state.community_cards.append(draw_card(next_state))
                    next_state.chips_already_call = [0] * len(self.players)
                    next_state.chips_needed_to_call = 0
                    next_state.minimum_bet = 20
                    next_state.ready_next_round = False
                    next_state.round = GameRound.TURN
                else:
                    next_state.index = (next_state.index + 1) % len(self.players)
                result = self.update_state(next_state)
                if training_mode:
                    for record in self.records:
                        if record[0] == state and record[2] == next_state:
                            record[3] = result
                expected_value = array_add(expected_value, result)
            return array_div(expected_value, len(next_states))
        elif state.round == GameRound.TURN:
            for next_state in next_states:
                if next_state.index == next_state.dealer:
                    next_state.ready_next_round = True
                if check_can_enter_next_round(next_state):
                    calculate_max_win(next_state)
                    next_state.index = (next_state.dealer + 1) % len(self.players)
                    next_state.community_cards.append(draw_card(next_state))
                    next_state.chips_already_call = [0] * len(self.players)
                    next_state.chips_needed_to_call = 0
                    next_state.minimum_bet = 20
                    next_state.ready_next_round = False
                    next_state.round = GameRound.RIVER
                else:
                    next_state.index = (next_state.index + 1) % len(self.players)
                result = self.update_state(next_state)
                if training_mode:
                    for record in self.records:
                        if record[0] == state and record[2] == next_state:
                            record[3] = result
                expected_value = array_add(expected_value, result)
            return array_div(expected_value, len(next_states))
        elif state.round == GameRound.RIVER:
            for next_state in next_states:
                if next_state.index == next_state.dealer:
                    next_state.ready_next_round = True
                if check_can_enter_next_round(next_state):
                    calculate_max_win(next_state)
                    next_state.chips_already_call = [0] * len(self.players)
                    result = settle(next_state)
                    expected_value = array_add(expected_value, result)
                    expected_value = array_sub(expected_value, next_state.total_bet)
                    if training_mode:
                        for record in self.records:
                            if record[0] == state and record[2] == next_state:
                                record[3] = result
                else:
                    next_state.index = (next_state.index + 1) % len(self.players)
                    result = self.update_state(next_state)
                    if training_mode:
                        for record in self.records:
                            if record[0] == state and record[2] == next_state:
                                record[3] = result
                    expected_value = array_add(expected_value, result)
            return array_div(expected_value, len(next_states))
