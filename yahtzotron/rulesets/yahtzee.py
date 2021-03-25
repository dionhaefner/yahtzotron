"""
Official rules according to

https://www.hasbro.com/common/instruct/yahtzee.pdf
"""

from .base import make_category, Ruleset


def _sum_of_all_dice(roll):
    return sum(die_value * num_dice for die_value, num_dice in enumerate(roll, 1))


def _yahtzee_bonus(roll, filled_categories, scores):
    if 5 in roll and scores[-2] > 0:
        return 100

    return 0


@make_category(counts_towards_bonus=True)
def ones(roll, filled_categories, scores):
    return 1 * roll[0] + _yahtzee_bonus(roll, filled_categories, scores)


@make_category(counts_towards_bonus=True)
def twos(roll, filled_categories, scores):
    return 2 * roll[1] + _yahtzee_bonus(roll, filled_categories, scores)


@make_category(counts_towards_bonus=True)
def threes(roll, filled_categories, scores):
    return 3 * roll[2] + _yahtzee_bonus(roll, filled_categories, scores)


@make_category(counts_towards_bonus=True)
def fours(roll, filled_categories, scores):
    return 4 * roll[3] + _yahtzee_bonus(roll, filled_categories, scores)


@make_category(counts_towards_bonus=True)
def fives(roll, filled_categories, scores):
    return 5 * roll[4] + _yahtzee_bonus(roll, filled_categories, scores)


@make_category(counts_towards_bonus=True)
def sixes(roll, filled_categories, scores):
    return 6 * roll[5] + _yahtzee_bonus(roll, filled_categories, scores)


@make_category
def three_of_a_kind(roll, filled_categories, scores):
    if 5 in roll:
        return _sum_of_all_dice(roll) + _yahtzee_bonus(roll, filled_categories, scores)

    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 3:
            return 3 * die_value

    return 0


@make_category
def four_of_a_kind(roll, filled_categories, scores):
    if 5 in roll:
        return _sum_of_all_dice(roll) + _yahtzee_bonus(roll, filled_categories, scores)

    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 4:
            return 4 * die_value

    return 0


@make_category
def full_house(roll, filled_categories, scores):
    if 5 in roll:
        return 25 + _yahtzee_bonus(roll, filled_categories, scores)

    if 2 in roll and 3 in roll:
        return 25

    return 0


@make_category
def small_straight(roll, filled_categories, scores):
    if 5 in roll:
        return 30 + _yahtzee_bonus(roll, filled_categories, scores)

    for i in range(3):
        if all(roll[i : i + 4] >= (1, 1, 1, 1)):
            return 30

    return 0


@make_category
def large_straight(roll, filled_categories, scores):
    if 5 in roll:
        return 40 + _yahtzee_bonus(roll, filled_categories, scores)

    for i in range(2):
        if all(roll[i : i + 5] >= (1, 1, 1, 1, 1)):
            return 40

    return 0


@make_category
def yahtzee(roll, filled_categories, scores):
    if 5 in roll:
        return 50

    return 0


@make_category
def chance(roll, filled_categories, scores):
    return _sum_of_all_dice(roll) + _yahtzee_bonus(roll, filled_categories, scores)


yahtzee_rules = Ruleset(
    ruleset_name="yahtzee",
    num_dice=5,
    categories=(
        ones,
        twos,
        threes,
        fours,
        fives,
        sixes,
        three_of_a_kind,
        four_of_a_kind,
        full_house,
        small_straight,
        large_straight,
        yahtzee,
        chance,
    ),
    bonus_cutoff=63,
    bonus_score=35,
)
