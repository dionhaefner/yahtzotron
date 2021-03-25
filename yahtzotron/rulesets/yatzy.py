from .base import make_category, Ruleset


@make_category(counts_towards_bonus=True)
def ones(roll, filled_categories, scores):
    return 1 * roll[0]


@make_category(counts_towards_bonus=True)
def twos(roll, filled_categories, scores):
    return 2 * roll[1]


@make_category(counts_towards_bonus=True)
def threes(roll, filled_categories, scores):
    return 3 * roll[2]


@make_category(counts_towards_bonus=True)
def fours(roll, filled_categories, scores):
    return 4 * roll[3]


@make_category(counts_towards_bonus=True)
def fives(roll, filled_categories, scores):
    return 5 * roll[4]


@make_category(counts_towards_bonus=True)
def sixes(roll, filled_categories, scores):
    return 6 * roll[5]


@make_category
def one_pair(roll, filled_categories, scores):
    best_pair = 0
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 2:
            best_pair = die_value

    return best_pair * 2


@make_category
def two_pairs(roll, filled_categories, scores):
    best_pairs = [0, 0]
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 2:
            best_pairs[1] = best_pairs[0]
            best_pairs[0] = die_value

    if best_pairs[1] == 0:
        return 0

    return 2 * best_pairs[0] + 2 * best_pairs[1]


@make_category
def three_of_a_kind(roll, filled_categories, scores):
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 3:
            return 3 * die_value

    return 0


@make_category
def four_of_a_kind(roll, filled_categories, scores):
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 4:
            return 4 * die_value

    return 0


@make_category
def full_house(roll, filled_categories, scores):
    doublet = None
    triplet = None

    for die_value, num_dice in enumerate(roll, 1):
        if num_dice == 2:
            doublet = die_value
        elif num_dice == 3:
            triplet = die_value

    if doublet is None or triplet is None:
        return 0

    return 3 * triplet + 2 * doublet


@make_category
def small_straight(roll, filled_categories, scores):
    if all(roll[:5] >= (1, 1, 1, 1, 1)):
        return 15

    return 0


@make_category
def large_straight(roll, filled_categories, scores):
    if all(roll[1:] >= (1, 1, 1, 1, 1)):
        return 25

    return 0


@make_category
def chance(roll, filled_categories, scores):
    return sum(die_value * num_dice for die_value, num_dice in enumerate(roll, 1))


@make_category
def yatzy(roll, filled_categories, scores):
    if 5 in roll:
        return 50

    return 0


yatzy_rules = Ruleset(
    ruleset_name="yatzy",
    num_dice=5,
    categories=(
        ones,
        twos,
        threes,
        fours,
        fives,
        sixes,
        one_pair,
        two_pairs,
        three_of_a_kind,
        four_of_a_kind,
        full_house,
        small_straight,
        large_straight,
        chance,
        yatzy,
    ),
    bonus_cutoff=63,
    bonus_score=50,
)
