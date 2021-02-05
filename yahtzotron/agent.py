import os
import pickle
import functools

from loguru import logger
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

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
        player_scorecard_idx = slice(sum(input_shapes[:2]), sum(input_shapes[:3]))
        init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")

        x = hk.Linear(256, w_init=init)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(256, w_init=init)(x)
        x = jax.nn.relu(x)

        out_value = hk.Linear(1)(x)

        out_category = hk.Linear(num_categories)(x)
        out_category = jnp.where(
            (inputs[..., 0, None] == 0) & (inputs[..., player_scorecard_idx] == 1),
            -jnp.inf,
            out_category,
        )

        return out_category, jnp.squeeze(out_value, axis=-1)

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


def turn_fast(
    player_scorecard,
    opponent_scorecard,
    objective,
    net,
    weights,
    num_dice,
    num_categories,
    roll_lut,
    greedy=False,
):
    player_scorecard_arr = player_scorecard.to_array()

    if objective == "win":
        opponent_scorecard_arr = opponent_scorecard.to_array()

    elif objective == "avg_score":
        # beating someone with equal score means maximizing expected final score
        opponent_scorecard_arr = player_scorecard_arr

    current_dice = (0,) * num_dice
    dice_to_keep = (0,) * num_dice

    for rolls_left in range(2, -1, -1):
        current_dice = tuple(
            sorted(
                die if keep else np.random.randint(1, 7)
                for die, keep in zip(current_dice, dice_to_keep)
            )
        )
        dice_count = np.bincount(current_dice, minlength=7)[1:]

        net_input, category_idx, value = get_action(
            rolls_left,
            dice_count,
            player_scorecard_arr,
            opponent_scorecard_arr,
            net,
            weights,
        )

        if greedy:
            category_idx = get_action_greedy(
                rolls_left, current_dice, player_scorecard, roll_lut
            )

        if rolls_left > 0:
            dice_to_keep = max(
                roll_lut["full"][current_dice][category_idx].keys(),
                key=lambda k: roll_lut["full"][current_dice][category_idx][k],
            )
        else:
            dice_to_keep = (1,) * num_dice

        logger.debug(" Observation: {}", net_input)
        logger.debug(" Cat. action: {}", category_idx)
        logger.debug(" Keep action: {}", dice_to_keep)
        logger.debug(" Value: {}", value)

        yield dict(
            rolls_left=rolls_left,
            net_input=net_input,
            category_idx=category_idx,
            value=value,
            dice_count=dice_count,
        )

    logger.info("Final roll: {} | Picked category: {}", current_dice, category_idx)


def get_action(
    rolls_left,
    dice_count,
    player_scorecard,
    opponent_scorecard,
    net,
    weights,
):
    def choose_from_logits(logits):
        # pure NumPy version of jax.random.categorical
        logits = np.asarray(logits)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        return np.random.choice(logits.shape[0], p=prob)

    network_inputs = np.concatenate(
        [
            np.array([rolls_left]),
            np.array(dice_count),
            player_scorecard,
            opponent_scorecard,
        ]
    )
    logits, value = net(weights, network_inputs)
    category_action = choose_from_logits(logits)

    return network_inputs, category_action, value


def get_action_greedy(rolls_left, current_dice, player_scorecard, roll_lut):
    # greedily pick action with highest expected reward advantage
    # this is not optimal play but should be a good baseline
    num_dice = player_scorecard.ruleset_.num_dice
    num_categories = player_scorecard.ruleset_.num_categories

    if rolls_left > 0:
        best_payoff = lambda lut, cat: max(lut[cat].values())
        marginal_lut = roll_lut["marginal-1"]
    else:
        # there is no keep action, so only keeping all dice counts
        best_payoff = lambda lut, cat: lut[cat][(1,) * num_dice]
        marginal_lut = roll_lut["marginal-0"]

    expected_payoff = [
        (
            best_payoff(roll_lut["full"][current_dice], c)
            if player_scorecard.filled[c] == 0
            else -float("inf")
        )
        - marginal_lut[c]
        for c in range(num_categories)
    ]
    category_action = np.argmax(expected_payoff)

    return category_action


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
        opponent_scorecard=None,
        pretrain=False,
    ):
        if self._objective == "win" and opponent_scorecard is None:
            raise ValueError('Opponent scorecard must be given for "win" objective')

        roll_lut = get_lut(self._roll_lut_path, self._ruleset)

        yield from turn_fast(
            player_scorecard,
            opponent_scorecard,
            objective=self._objective,
            net=self._net,
            weights=self._weights,
            num_dice=self._ruleset.num_dice,
            num_categories=self._ruleset.num_categories,
            roll_lut=roll_lut,
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
