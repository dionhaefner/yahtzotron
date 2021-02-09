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
def create_network(objective, num_dice, num_categories):
    input_shapes = [
        1,  # number of rerolls left
        6,  # count of each die value
        num_categories,  # player scorecard
        2,  # player upper and lower scores
    ]

    if objective == "win":
        input_shapes.append(
            1,  # opponent value
        )

    keep_action_space = 2 ** num_dice

    def network(inputs):
        player_scorecard_idx = slice(sum(input_shapes[:2]), sum(input_shapes[:3]))
        init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")

        x = hk.Linear(256, w_init=init)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(256, w_init=init)(x)
        x = jax.nn.relu(x)

        out_value = hk.Linear(1)(x)

        dice_encoding = hk.Linear(num_dice)(x)
        dice_encoding = jax.nn.sigmoid(dice_encoding)
        out_keep = hk.Linear(keep_action_space)(dice_encoding)

        out_category = hk.Linear(num_categories)(x)
        out_category = jnp.where(
            # disallow already filled categories
            inputs[..., player_scorecard_idx] == 1,
            -jnp.inf,
            out_category,
        )

        return out_keep, out_category, jnp.squeeze(out_value, axis=-1)

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


def play_turn(
    player_scorecard,
    objective,
    net,
    weights,
    num_dice,
    num_categories,
    roll_lut,
    opponent_value=None,
    greedy=False,
):
    player_scorecard_arr = player_scorecard.to_array()

    if opponent_value is None and objective == "win":
        raise ValueError("opponent value must be given for win objective")

    current_dice = (0,) * num_dice
    dice_to_keep = (0,) * num_dice

    for rolls_left in range(2, -1, -1):
        kept_dice = tuple(die for die, keep in zip(current_dice, dice_to_keep) if keep)
        num_dice_to_roll = num_dice - len(kept_dice)
        roll_input = tuple((yield num_dice_to_roll))
        current_dice = tuple(sorted(kept_dice + roll_input))
        dice_count = np.bincount(current_dice, minlength=7)[1:]

        if greedy:
            net_input = assemble_network_inputs(
                rolls_left, dice_count, player_scorecard_arr, opponent_value
            )
            keep_action, category_idx = get_action_greedy(
                rolls_left, current_dice, player_scorecard, roll_lut
            )
            value = None
        else:
            net_input, keep_action, category_idx, value = get_action(
                rolls_left,
                dice_count,
                player_scorecard_arr,
                opponent_value,
                net,
                weights,
            )

        if rolls_left > 0:
            dice_to_keep = np.unpackbits(np.uint8(keep_action), count=num_dice, bitorder='little')
        else:
            dice_to_keep = (1,) * num_dice

        logger.debug(" Observation: {}", net_input)
        logger.debug(" Cat. action: {}", category_idx)
        logger.debug(" Keep action: {}", dice_to_keep)
        logger.debug(" Value: {}", value)

        yield dict(
            rolls_left=rolls_left,
            net_input=net_input,
            keep_action=keep_action,
            category_idx=category_idx,
            value=value,
            dice_count=dice_count,
            dice_to_keep=dice_to_keep,
        )

    cat_name = player_scorecard.ruleset_.categories[category_idx].name
    logger.info(
        "Final roll: {} | Picked category: {} ({})",
        current_dice,
        category_idx,
        cat_name,
    )


def assemble_network_inputs(
    rolls_left, dice_count, player_scorecard, opponent_value=None
):
    inputs = [
        np.asarray([rolls_left]),
        dice_count,
        player_scorecard,
    ]
    if opponent_value is not None:
        inputs.append(np.asarray([opponent_value]))

    return np.concatenate(inputs)


def get_action(
    rolls_left,
    dice_count,
    player_scorecard,
    opponent_value,
    network,
    weights,
):
    def choose_from_logits(logits):
        # pure NumPy version of jax.random.categorical
        logits = np.asarray(logits)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        return np.random.choice(logits.shape[0], p=prob)

    network_inputs = assemble_network_inputs(
        rolls_left, dice_count, player_scorecard, opponent_value
    )
    keep_logits, category_logits, value = network(weights, network_inputs)
    keep_action = choose_from_logits(keep_logits)
    category_action = choose_from_logits(category_logits)

    return network_inputs, keep_action, category_action, value


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

    if rolls_left > 0:
        dice_to_keep = max(
            roll_lut["full"][current_dice][category_action].keys(),
            key=lambda k: roll_lut["full"][current_dice][category_action][k],
        )
    else:
        dice_to_keep = (1,) * num_dice

    keep_action = int(np.packbits(dice_to_keep, bitorder='little'))
    return keep_action, category_action


class Yahtzotron:
    def __init__(self, ruleset=None, load_path=None, objective="win", greedy=False):
        if ruleset is None and load_path is None:
            raise ValueError("ruleset must be given")

        possible_objectives = ("win", "avg_score")
        if objective not in possible_objectives:
            raise ValueError(
                f"Got unexpected objective {objective}, must be one of {possible_objectives}"
            )

        if load_path is None:
            self._ruleset = AVAILABLE_RULESETS[ruleset]
            self._objective = objective
        else:
            self.load(load_path)

        self._default_greedy = greedy
        self._roll_lut_path = os.path.join(
            DISK_CACHE, f"roll_lut_{self._ruleset.name}.pkl"
        )

        num_dice, num_categories = (
            self._ruleset.num_dice,
            self._ruleset.num_categories,
        )

        net, input_shape = create_network(self._objective, num_dice, num_categories)
        self._network = jax.jit(net.apply)

        if load_path is None:
            self._weights = net.init(
                next(key), jnp.ones((1, sum(input_shape)), dtype=jnp.float32)
            )

    def turn(
        self,
        player_scorecard,
        opponent_scorecards=None,
        greedy=None,
    ):
        if greedy is None:
            greedy = self._default_greedy

        if self._objective == "win":
            if opponent_scorecards is None:
                raise ValueError(
                    'Opponent scorecards must be given for "win" objective'
                )

            if not hasattr(opponent_scorecards, "__iter__"):
                opponent_scorecards = [opponent_scorecards]

            net_input = np.stack(
                [
                    assemble_network_inputs(2, np.zeros(6), o.to_array(), 0.0)
                    for o in opponent_scorecards
                ],
                axis=0,
            )
            _, _, opponent_values = self._network(self._weights, net_input)
            # print(opponent_values)
            opponent_value = np.max(opponent_values)
        else:
            opponent_value = None

        if greedy:
            roll_lut = get_lut(self._roll_lut_path, self._ruleset)
        else:
            roll_lut = None

        yield from play_turn(
            player_scorecard,
            opponent_value=opponent_value,
            objective=self._objective,
            net=self._network,
            weights=self._weights,
            num_dice=self._ruleset.num_dice,
            num_categories=self._ruleset.num_categories,
            roll_lut=roll_lut,
            greedy=greedy,
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
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)

        statedict = dict(
            objective=self._objective,
            ruleset=self._ruleset.name,
            weights=self._weights,
        )

        with open(path, "wb") as f:
            pickle.dump(statedict, f)

    def load(self, path):
        with open(path, "rb") as f:
            statedict = pickle.load(f)

        self._objective = statedict["objective"]
        self._ruleset = AVAILABLE_RULESETS[statedict["ruleset"]]
        self._weights = statedict["weights"]

    def __repr__(self):
        return f"{self.__class__.__name__}(ruleset={self._ruleset}, objective={self._objective})"
