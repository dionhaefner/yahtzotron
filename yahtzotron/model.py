import os
import pickle
import functools

import jax
import jax.numpy as jnp
import haiku as hk

from .play import turn_fast
from .rulesets import AVAILABLE_RULESETS
from .strategy import assemble_roll_lut

key = hk.PRNGSequence(17)
memoize = functools.lru_cache(maxsize=None)

DISK_CACHE = os.path.expanduser(os.path.join("~", ".yahtzotron"))


@memoize
def create_network(num_dice, num_categories):
    input_shapes = [
        1,  # current roll number
        6,  # count of each die value
        num_categories,  # player scorecard
        2,  # player scores
        num_categories,  # opponent scorecard
        2,  # opponent scores
    ]

    def network(inputs):
        init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")

        x = hk.Linear(256, w_init=init)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(256, w_init=init)(x)
        x = jax.nn.relu(x)

        out_value = hk.Linear(1)(x)
        out_policy = hk.Linear(num_categories)(x)
        out_policy = jnp.where(
            inputs[..., 7 : 7 + num_categories] == 1, -jnp.inf, out_policy
        )

        return out_policy, jnp.squeeze(out_value, axis=-1)

    forward = hk.without_apply_rng(hk.transform(network))
    return forward, input_shapes


@memoize
def get_lut(path, ruleset):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        roll_lut = assemble_roll_lut(ruleset)
        with open(path, "wb") as f:
            pickle.dump(roll_lut, f)

    with open(path, "rb") as f:
        roll_lut = pickle.load(f)

    return roll_lut


class Yahtzotron:
    def __init__(self, ruleset, load_path=None, objective="win"):
        self._ruleset = AVAILABLE_RULESETS[ruleset]
        self._roll_lut_path = os.path.join(DISK_CACHE, f"roll_lut_{ruleset}.pkl")

        num_dice, num_categories = (
            self._ruleset.num_dice,
            self._ruleset.num_categories,
        )

        net, input_shape = create_network(num_dice, num_categories)
        self._net = jax.jit(net.apply)
        self._weights = net.init(
            next(key), jnp.ones((1, sum(input_shape)), dtype=jnp.float32)
        )

        possible_objectives = ("win", "avg_score")
        if objective not in possible_objectives:
            raise ValueError(
                f"Got unexpected objective {objective}, must be one of {possible_objectives}"
            )

        self._objective = objective

        if load_path is not None:
            self.load(load_path)

    def turn(
        self,
        player_scorecard,
        opponent_scorecards,
        return_all_actions=False,
        pretrain=False,
    ):
        roll_lut = get_lut(self._roll_lut_path, self._ruleset)

        return turn_fast(
            player_scorecard,
            opponent_scorecards,
            objective=self._objective,
            net=self._net,
            weights=self._weights,
            num_dice=self._ruleset.num_dice,
            num_categories=self._ruleset.num_categories,
            roll_lut=roll_lut,
            return_all_actions=return_all_actions,
            greedy=pretrain,
        )

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

        self._objective = statedict["objective"]
        self._ruleset = AVAILABLE_RULESETS[statedict["ruleset"]]
        self._weights = statedict["weights"]

    def __repr__(self):
        return f"{self.__class__.__name__}(ruleset={self._ruleset}, objective={self._objective})"
