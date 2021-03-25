from collections import namedtuple


Category = namedtuple("category", ["name", "score", "counts_towards_bonus"])


def make_category(*args, name=None, counts_towards_bonus=False):
    """Decorator that creates a new category from a function."""

    def inner(func):
        cat_name = name or func.__name__
        return Category(cat_name, func, counts_towards_bonus)

    if args and callable(args[0]):
        return inner(args[0])

    return inner


class Ruleset:
    """Represents the rules of the game.

    Used to convert rolls and categories to scores, and to compute total scores.
    """

    def __init__(
        self,
        categories,
        num_dice=5,
        bonus_cutoff=63,
        bonus_score=50,
        ruleset_name="custom",
    ):
        self.name = ruleset_name
        self.num_dice = num_dice
        self.num_rounds = len(categories)
        self.num_categories = len(categories)
        self.categories = categories

        self.bonus_cutoff_ = bonus_cutoff
        self.bonus_score_ = bonus_score

    def score(self, roll, cat_idx, filled_categories, scores):
        if filled_categories[cat_idx]:
            raise ValueError("Cannot score already filled category")
        return self.categories[cat_idx].score(roll, filled_categories, scores)

    def total_score(self, scores):
        return scores.sum() + self.bonus_value(scores)

    def score_summary(self, scores):
        upper_score = 0
        lower_score = 0

        for cat, score in zip(self.categories, scores):
            if cat.counts_towards_bonus:
                upper_score += score
            else:
                lower_score += score

        return upper_score, lower_score

    def bonus_value(self, scores):
        upper_score, _ = self.score_summary(scores)

        if upper_score >= self.bonus_cutoff_:
            return self.bonus_score_

        return 0

    def __repr__(self):
        return f"{self.__class__.__name__}(ruleset_name={self.name})"
