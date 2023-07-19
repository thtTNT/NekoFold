import time

from game import Game
from player import Player, AIPlayer


def main():
    it = 0
    score = 0
    while it < 1024:
        ai_player = AIPlayer(training_mode=False)
        game = Game(players=[ai_player, Player()],
                    initial_chips=[2000, 2000])
        game_result, records = game.play()
        score += game_result[0]
        print(score)
        ai_player.train(records)
        it += 1
        # ai_player.save(it)
        print(f"iteration {it} finished")


main()
