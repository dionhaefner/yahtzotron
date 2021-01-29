import os
import pickle
import functools

import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

from .play import turn_fast
from .rulesets import AVAILABLE_RULESETS
from .strategy import assemble_roll_lut

key = hk.PRNGSequence(17)
memoize = functools.lru_cache(maxsize=None)

DISK_CACHE = os.path.expanduser(os.path.join('~', '.yahtzotron'))


@memoize
def create_strategy_net(num_dice, num_categories):
    """Predicts category picked after turn"""

    input_shapes = [
        1,  # current roll number
        6,  # count of each die value
        num_categories,  # player scorecard
        2,  # player scores
        num_categories,  # opponent scorecard
        2,  # opponent scores
    ]

    def model(x):
        m = hk.Sequential(
            [
                hk.Linear(12),
                jax.nn.relu,
                hk.Linear(24),
                jax.nn.relu,
                hk.Linear(12),
                jax.nn.relu,
                hk.Linear(num_categories),
                jax.nn.softmax,
            ]
        )
        return m(x)

    forward = hk.without_apply_rng(hk.transform(model))
    return forward, input_shapes


@memoize
def create_value_net(num_categories):
    """Predicts probability of player 1 winning over player 2"""

    input_shapes = [
        num_categories,  # player scorecard
        2,  # player scores
        num_categories,  # opponent scorecard
        2,  # opponent scores
    ]

    def model(x):
        m = hk.Sequential(
            [
                hk.Linear(12),
                jax.nn.relu,
                hk.Linear(24),
                jax.nn.relu,
                hk.Linear(12),
                jax.nn.relu,
                hk.Linear(1),
                jax.nn.sigmoid,
            ]
        )
        return m(x)

    forward = hk.without_apply_rng(hk.transform(model))
    return forward, input_shapes


class Yahtzotron:
    def __init__(self, ruleset, load_path=None, objective="win"):
        self._ruleset = AVAILABLE_RULESETS[ruleset]
        self._roll_lut_path = os.path.join(DISK_CACHE, f'roll_lut_{ruleset}.pkl')

        num_dice, num_categories = (
            self._ruleset.num_dice,
            self._ruleset.num_categories,
        )
        nets_and_shapes = dict(
            # roll=create_roll_net(num_dice, num_categories),
            strategy=create_strategy_net(num_dice, num_categories),
            value=create_value_net(num_categories),
        )
        self._nets = {k: v[0] for k, v in nets_and_shapes.items()}
        self._weights = {k: v[0].init(next(key), jnp.ones([1, np.sum(v[1])])) for k, v in nets_and_shapes.items()}

        possible_objectives = ("win", "avg_score")
        if objective not in possible_objectives:
            raise ValueError(
                f"Got unexpected objective {objective}, must be one of {possible_objectives}"
            )

        self._objective = objective
        self._reinit_model = True

        if load_path is not None:
            self.load(load_path)

    def turn(self, player_scorecard, opponent_scorecards, return_all_actions=False):
        if not os.path.isfile(self._roll_lut_path):
            os.makedirs(os.path.dirname(self._roll_lut_path), exist_ok=True)
            roll_lut = assemble_roll_lut(self._ruleset)
            with open(self._roll_lut_path, 'wb') as f:
                pickle.dump(roll_lut, f)

        with open(self._roll_lut_path, 'rb') as f:
            roll_lut = pickle.load(f)

        return turn_fast(
            player_scorecard,
            opponent_scorecards,
            objective=self._objective,
            nets=self._nets,
            weights=self._weights,
            num_dice=self._ruleset.num_dice,
            num_categories=self._ruleset.num_categories,
            roll_lut=roll_lut,
        )

    def pre_train(self, num_samples=100_000):
        """Pre-train value network to go for maximum scores"""
        return
        # train_scorecard = np.random.random_integers(0, 1, size=(num_samples, self._ruleset.num_categories))
        # train_scores = np.random.random_integers(0, 125, size=(2, num_samples, 2))
        # winning_log_odds = (train_scores[0].sum(axis=1) - train_scores[1].sum(axis=1)) / 50
        # winning_prob = 1. / (1. + np.exp(-winning_log_odds))
        # train_labels = np.random.binomial(1, p=winning_prob, size=(num_samples,))

        # self._nets['value'].fit(
        #     [train_scorecard, train_scores[0], train_scorecard, train_scores[1]],
        #     train_labels,
        #     epochs=2
        # )

        # train_rolls = np.random.random_integers(1, 6, size=(num_samples, self._ruleset.num_dice))
        # train_dice_counts = np.apply_along_axis(
        #     lambda x: np.bincount(x, minlength=7)[1:],
        #     axis=1, arr=train_rolls
        # )

        # train_rewards = np.array([
        #     [self._ruleset.score(dice_count, cat_idx, scorecard) if not scorecard[cat_idx] else -1 for cat_idx in range(self._ruleset.num_categories)]
        #     for dice_count, scorecard in zip(train_dice_counts, train_scorecard)
        # ])

        # train_best_category = to_categorical(np.argmax(train_rewards, axis=-1))

        # self._nets['strategy_1'].fit(
        #     [train_dice_counts, train_scorecard, train_scores[0], train_scorecard, train_scores[1]],
        #     train_best_category,
        #     epochs=10
        # )

        # self._nets['strategy_2'].fit(
        #     [train_dice_counts, train_scorecard, train_scores[0], train_scorecard, train_scores[1]],
        #     train_best_category,
        #     epochs=10
        # )

    def get_weights(self):
        return self._weights

    def set_weights(self, new_weights):
        self._weights = hk.data_structures.to_immutable_dict(new_weights)

    def clone(self, keep_weights=True):
        yzt = self.__class__(ruleset=self._ruleset.name, objective=self._objective)

        if keep_weights:
            yzt.set_weights(self.get_weights())

        return yzt

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        statedict = dict(
            objective=self._objective,
            ruleset=self._ruleset.name,
            weights=self._weights,
        )

        with open(os.path.join(path, "yzt.pkl"), "wb") as f:
            pickle.dump(statedict, f)

    def load(self, path):
        with open(os.path.join(path, "yzt.pkl"), "rb") as f:
            statedict = pickle.load(f)

        self._objective = statedict['objective']
        self._ruleset = AVAILABLE_RULESETS[statedict['ruleset']]
        self._weights = statedict['weights']

    def __repr__(self):
        return f"{self.__class__.__name__}(ruleset={self._ruleset}, objective={self._objective})"
