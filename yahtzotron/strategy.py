from math import factorial
from functools import lru_cache
import itertools

import tqdm
import numpy as np


@lru_cache(maxsize=None)
def _memoized_score(roll, cat_idx, scorecard, ruleset):
    dice_count = np.bincount(roll, minlength=7)[1:]
    return ruleset.score(dice_count, cat_idx, scorecard)


def get_expected_reward(initial_roll, cat_idx, ruleset):
    num_dice = ruleset.num_dice
    dummy_scorecard = (0,) * ruleset.num_categories

    expected_reward = [0.] * (2 ** num_dice)
    for i, action in enumerate(itertools.product((0, 1), repeat=num_dice)):
        kept_dice = tuple(die for die, kept in zip(initial_roll, action) if kept)
        num_rerolled = num_dice - len(kept_dice)

        count = 0
        for final_roll in itertools.combinations_with_replacement(range(1, 7), num_rerolled):
            final_dice = tuple(sorted(kept_dice + final_roll))
            expected_reward[i] += _memoized_score(final_dice, cat_idx, dummy_scorecard, ruleset)
            count += 1

        expected_reward[i] /= count

    return expected_reward


def assemble_roll_lut(ruleset):
    num_dice = ruleset.num_dice

    # all possible unique rolls with n 6-sided dice
    roll_combinations = itertools.combinations_with_replacement(range(1, 7), num_dice)
    total_elements = int(factorial(5 + num_dice) / factorial(num_dice) / factorial(5))

    expected_reward = {}

    pbar = tqdm.tqdm(roll_combinations, desc='🎲🎲🎲', total=total_elements)

    for initial_roll in pbar:
        for cat_idx in range(ruleset.num_categories):
            expected_reward[(initial_roll, cat_idx)] = get_expected_reward(
                initial_roll, cat_idx, ruleset
            )

    return expected_reward


def reward_idx_to_action(idx, num_dice):
    return list(itertools.product((0, 1), repeat=num_dice))[idx]