import itertools


class Card:

    def __init__(self, name: str):
        if name[0] == "S":
            self.type = "Spade"
        elif name[0] == "H":
            self.type = "Heart"
        elif name[0] == "D":
            self.type = "Diamond"
        elif name[0] == "C":
            self.type = "Club"
        if name[1] == "A":
            self.value = 14
        elif name[1] == "K":
            self.value = 13
        elif name[1] == "Q":
            self.value = 12
        elif name[1] == "J":
            self.value = 11
        elif name[1] == "T":
            self.value = 10
        else:
            self.value = int(name[1])


def findRoyalFlush(cards: list[Card]):
    combs = itertools.combinations(cards, 5)
    for comb in combs:
        sorted_cards = sorted(comb, key=lambda x: x.value)
        if sorted_cards[0].value == 10 and sorted_cards[1].value == 11 and sorted_cards[2].value == 12 \
                and sorted_cards[3].value == 13 and sorted_cards[4].value == 14 \
                and cards[0].type == cards[1].type == cards[2].type == cards[3].type == cards[4].type:
            return "RoyalFlush", []
    return None


def findStraightFlush(cards: list[Card]):
    combs = itertools.combinations(cards, 5)
    result = None
    for comb in combs:
        sorted_cards = sorted(comb, key=lambda x: x.value)
        if ((sorted_cards[0].value == sorted_cards[1].value - 1 and
             sorted_cards[1].value == sorted_cards[2].value - 1 and
             sorted_cards[2].value == sorted_cards[3].value - 1 and
             sorted_cards[3].value == sorted_cards[4].value - 1)
            or (sorted_cards[0].value == 2 and sorted_cards[1].value == 3 and sorted_cards[2].value == 4 and
                sorted_cards[3].value == 5 and sorted_cards[4].value == 14)) \
                and cards[0].type == cards[1].type == cards[2].type == cards[3].type == cards[4].type:
            if sorted_cards[2].value == 0:
                new_result = ("StraightFlush", [5])
            else:
                new_result = ("StraightFlush", [sorted_cards[4].value])
            if result is None or new_result[1] > result[1]:
                result = new_result
    return result


def findFourOfAKind(cards: list[Card]):
    combs = itertools.combinations(cards, 5)
    for comb in combs:
        for ignore_card in comb:
            new_cards = list(comb).copy()
            new_cards.remove(ignore_card)
            if new_cards[0].value == new_cards[1].value == new_cards[2].value == new_cards[3].value:
                return "FourOfAKind", [new_cards[0].value, ignore_card.value]
    return None


def findFullHouse(cards: list[Card]):
    result = None
    combs = itertools.combinations(cards, 5)
    for comb in combs:
        sorted_cards = sorted(comb, key=lambda x: x.value)
        if sorted_cards[0].value == sorted_cards[1].value == sorted_cards[2].value \
                and sorted_cards[3].value == sorted_cards[4].value:
            if result is None:
                result = ("FullHouse", [sorted_cards[0].value, sorted_cards[3].value])
            else:
                new_result = ("FullHouse", [sorted_cards[0].value, sorted_cards[3].value])
                if new_result[1] > result[1]:
                    result = new_result
        elif sorted_cards[0].value == sorted_cards[1].value \
                and sorted_cards[2].value == sorted_cards[3].value == sorted_cards[4].value:
            if result is None or result[1] < [sorted_cards[2].value, sorted_cards[0].value]:
                result = ("FullHouse", [sorted_cards[2].value, sorted_cards[0].value])
    return result


def findFlush(cards: list[Card]):
    result = None
    combs = itertools.combinations(cards, 5)
    for comb in combs:
        sorted_cards = sorted(comb, key=lambda x: x.value)
        if sorted_cards[0].type == sorted_cards[1].type == \
                sorted_cards[2].type == sorted_cards[3].type == sorted_cards[4].type:
            if result is None or result[1] < [sorted_cards[4].value, sorted_cards[3].value, sorted_cards[2].value,
                                              sorted_cards[1].value, sorted_cards[0].value]:
                result = ("Flush", [sorted_cards[4].value, sorted_cards[3].value,
                                    sorted_cards[2].value, sorted_cards[1].value, sorted_cards[0].value])
    return result


def findStraight(cards: list[Card]):
    result = None
    combs = itertools.combinations(cards, 5)
    for comb in combs:
        sorted_cards = sorted(comb, key=lambda x: x.value)
        if (sorted_cards[0].value == sorted_cards[1].value - 1
            and sorted_cards[1].value == sorted_cards[2].value - 1
            and sorted_cards[2].value == sorted_cards[3].value - 1
            and sorted_cards[3].value == sorted_cards[4].value - 1) \
                or (sorted_cards[0].value == 2 and sorted_cards[1].value == 3 and sorted_cards[2].value == 4 and
                    sorted_cards[3].value == 5 and sorted_cards[4].value == 14):
            if sorted_cards[0].value == 2:
                new_result = ("Straight", [5])
            else:
                new_result = ("Straight", [sorted_cards[4].value])
            if result is None or new_result[1] > result[1]:
                result = new_result
    return result


def findThreeOfAKind(cards: list[Card]):
    result = None
    combs = itertools.combinations(cards, 3)
    for comb in combs:
        if comb[0].value == comb[1].value == comb[2].value:
            new_cards = list(cards).copy()
            new_cards.remove(comb[0])
            new_cards.remove(comb[1])
            new_cards.remove(comb[2])
            new_cards = sorted(new_cards, key=lambda x: x.value)
            if result is None or result[1] < [comb[0].value, new_cards[3].value, new_cards[2].value]:
                result = ("ThreeOfAKind", [comb[0].value, new_cards[3].value, new_cards[2].value])
    return result


def findTwoPairs(cards: list[Card]):
    result = None
    combs = itertools.combinations(cards, 4)
    for comb in combs:
        if comb[0].value == comb[1].value and comb[2].value == comb[3].value:
            new_cards = list(cards).copy()
            new_cards.remove(comb[0])
            new_cards.remove(comb[1])
            new_cards.remove(comb[2])
            new_cards.remove(comb[3])
            new_cards = sorted(new_cards, key=lambda x: x.value)
            if result is None or result[1] < [comb[2].value, comb[0].value, new_cards[2].value]:
                result = ("TwoPairs", [comb[2].value, comb[0].value, new_cards[2].value])
    return result


def findPair(cards: list[Card]):
    result = None
    combs = itertools.combinations(cards, 2)
    for comb in combs:
        if comb[0].value == comb[1].value:
            new_cards = list(cards).copy()
            new_cards.remove(comb[0])
            new_cards.remove(comb[1])
            new_cards = sorted(new_cards, key=lambda x: x.value)
            if result is None or result[1] < [comb[0].value, new_cards[3].value, new_cards[2].value,
                                              new_cards[1].value]:
                result = ("Pair", [comb[0].value, new_cards[3].value, new_cards[2].value, new_cards[1].value])
    return result


def findHighCard(cards: list[Card]):
    new_cards = sorted(cards, key=lambda x: x.value)
    return ("HighCard", [new_cards[6].value, new_cards[5].value, new_cards[4].value,
                         new_cards[3].value, new_cards[2].value])


def findPattern(card_strs: list[str]):
    cards = [Card(card_str) for card_str in card_strs]
    result = findRoyalFlush(cards)
    if result is not None:
        return result
    result = findStraightFlush(cards)
    if result is not None:
        return result
    result = findFourOfAKind(cards)
    if result is not None:
        return result
    result = findFullHouse(cards)
    if result is not None:
        return result
    result = findFlush(cards)
    if result is not None:
        return result
    result = findStraight(cards)
    if result is not None:
        return result
    result = findThreeOfAKind(cards)
    if result is not None:
        return result
    result = findTwoPairs(cards)
    if result is not None:
        return result
    result = findPair(cards)
    if result is not None:
        return result
    return findHighCard(cards)


PATTERN_ORDER = ["RoyalFlush", "StraightFlush", "FourOfAKind", "FullHouse", "Flush", "Straight", "ThreeOfAKind",
                 "TwoPairs", "Pair", "HighCard"]


def compare_pattern(pattern1, pattern2):
    if pattern1[0] != pattern2[0]:
        return -(PATTERN_ORDER.index(pattern1[0]) - PATTERN_ORDER.index(pattern2[0]))
    if pattern1[1] > pattern2[1]:
        return 1
    if pattern1[1] < pattern2[1]:
        return -1
    return 0


if __name__ == "__main__":
    print(findPattern("SA SK SQ SJ ST S9 S8".split(" ")))
    print(findPattern("S7 SK SQ SJ ST S9 S8".split(" ")))
    print(findPattern("S7 D7 C7 H7 S6 D6 H6".split(" ")))
    print(findPattern("S7 D5 C7 H6 S6 D7 H5".split(" ")))
    print(findPattern("SK SJ ST S6 S2 S3 S5".split(" ")))
    print(findPattern("DK SQ SJ ST C9 D8 S7".split(" ")))
    print(findPattern("DK SK SK SJ CT DA S7".split(" ")))
    print(findPattern("DJ SJ DK SK CT DT SA".split(" ")))
    print(findPattern("DJ SJ D5 SK CT D8 SA".split(" ")))
    print(findPattern("DQ SJ D5 SK C9 D8 SA".split(" ")))
