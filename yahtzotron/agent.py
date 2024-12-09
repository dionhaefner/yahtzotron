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
    """Create the neural networks used by the agent."""
    from yahtzotron.training import MINIMUM_LOGIT

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
        rolls_left = inputs[..., 0, None]
        player_scorecard_idx = slice(sum(input_shapes[:2]), sum(input_shapes[:3]))
        init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")

        x = hk.Linear(128, w_init=init)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(256, w_init=init)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(128, w_init=init)(x)
        x = jax.nn.relu(x)

        out_value = hk.Linear(1)(x)

        out_keep = hk.Linear(keep_action_space)(x)

        out_category = hk.Linear(num_categories)(x)
        out_category = jnp.where(
            # disallow already filled categories
            inputs[..., player_scorecard_idx] == 1,
            -jnp.inf,
            out_category,
        )

        def pad_action(logit, num_pad):
            pad_shape = [(0, 0)] * (logit.ndim - 1) + [(0, num_pad)]
            return jnp.pad(logit, pad_shape, constant_values=MINIMUM_LOGIT)

        if keep_action_space < num_categories:
            out_keep = pad_action(out_keep, num_categories - keep_action_space)

        elif keep_action_space > num_categories:
            out_category = pad_action(out_category, keep_action_space - num_categories)

        out_action = jnp.where(rolls_left == 0, out_category, out_keep)

        return out_action, jnp.squeeze(out_value, axis=-1)

    def strategy_network(inputs):
        player_scorecard_idx = slice(sum(input_shapes[:2]), sum(input_shapes[:3]))
        init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")

        x = hk.Linear(64, w_init=init)(inputs)
        x = jax.nn.relu(x)
        x = hk.Linear(128, w_init=init)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(64, w_init=init)(x)
        x = jax.nn.relu(x)

        out_category = hk.Linear(num_categories)(x)
        out_category = jnp.where(
            # disallow already filled categories
            inputs[..., player_scorecard_idx] == 1,
            MINIMUM_LOGIT,
            out_category,
        )

        return out_category

    forward = hk.without_apply_rng(hk.transform(network))
    forward_strategy = hk.without_apply_rng(hk.transform(strategy_network))

    return {
        "input-shapes": input_shapes,
        "network": forward,
        "strategy-network": forward_strategy,
    }


@memoize
def get_lut(path, ruleset):
    """Load cached look-up table, or compute from scratch."""
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        roll_lut = assemble_roll_lut(ruleset)
        with open(path, "wb") as f:
            pickle.dump(roll_lut, f)

    with open(path, "rb") as f:
        roll_lut = pickle.load(f)

    return roll_lut


def get_opponent_value(opponent_scorecards, network, weights):
    """Compute the maximum value of all given opponents."""
    if not hasattr(opponent_scorecards, "__iter__"):
        opponent_scorecards = [opponent_scorecards]

    net_input = np.stack(
        [
            assemble_network_inputs(2, np.zeros(6), o.to_array(), 0.0)
            for o in opponent_scorecards
        ],
        axis=0,
    )
    _, opponent_values = network(weights, net_input)
    return np.max(opponent_values)


def play_turn(
    player_scorecard,
    objective,
    network,
    weights,
    num_dice,
    use_lut=None,
    opponent_value=None,
):
    """Play a turn.

    This is a generator that yields a dict describing the current state
    after each action.
    """
    player_scorecard_arr = player_scorecard.to_array()

    if opponent_value is None and objective == "win":
        raise ValueError("opponent value must be given for win objective")

    current_dice = (0,) * num_dice
    dice_to_keep = (0,) * num_dice

    for rolls_left in (2, 1, 0):
        kept_dice = tuple(die for die, keep in zip(current_dice, dice_to_keep) if keep)
        roll_input = yield kept_dice
        current_dice = tuple(sorted(roll_input))
        dice_count = np.bincount(current_dice, minlength=7)[1:]
        assert np.all(dice_count - np.bincount(kept_dice, minlength=7)[1:] >= 0)

        if use_lut:
            observation = assemble_network_inputs(
                rolls_left, dice_count, player_scorecard_arr, opponent_value
            )
            action = get_action_greedy(
                rolls_left, current_dice, player_scorecard, use_lut
            )
            value = None
        else:
            observation, action, value = get_action(
                rolls_left,
                dice_count,
                player_scorecard_arr,
                opponent_value,
                network,
                weights,
            )

        if rolls_left > 0:
            keep_action = action
            category_idx = None
            dice_to_keep = np.unpackbits(
                np.uint8(keep_action), count=num_dice, bitorder="little"
            )
        else:
            keep_action = None
            category_idx = action
            dice_to_keep = [1] * num_dice

        logger.debug(" Observation: {}", observation)
        logger.debug(" Keep action: {}", dice_to_keep)
        logger.debug(" Value: {}", value)

        yield dict(
            rolls_left=rolls_left,
            observation=observation,
            action=action,
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
    """Re-shape inputs to be used as network input."""
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
    """Get an action according to current policy."""

    def choose_from_logits(logits):
        # pure NumPy version of jax.random.categorical
        logits = np.asarray(logits)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        return np.random.choice(logits.shape[0], p=prob)

    network_inputs = assemble_network_inputs(
        rolls_left, dice_count, player_scorecard, opponent_value
    )
    action_logits, value = network(weights, network_inputs)
    action = choose_from_logits(action_logits)

    return network_inputs, action, value


def get_action_greedy(rolls_left, current_dice, player_scorecard, roll_lut):
    """Get an action according to look-up table.

    This greedily picks the action with the highest expected reward advantage.
    This is far from optimal play but should be a good baseline.
    """
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
        keep_action = int(np.packbits(dice_to_keep, bitorder="little"))
        return keep_action

    return category_action


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

        self._be_greedy = greedy
        self._roll_lut_path = os.path.join(
            DISK_CACHE, f"roll_lut_{self._ruleset.name}.pkl"
        )

        num_dice, num_categories = (
            self._ruleset.num_dice,
            self._ruleset.num_categories,
        )

        networks = create_network(self._objective, num_dice, num_categories)
        self._network = jax.jit(networks["network"].apply)
        self._strategy_network = jax.jit(networks["strategy-network"].apply)

        if load_path is None:
            self._weights = networks["network"].init(
                next(key),
                jnp.ones((1, sum(networks["input-shapes"])), dtype=jnp.float32),
            )
            self._strategy_weights = networks["strategy-network"].init(
                next(key),
                jnp.ones((1, sum(networks["input-shapes"])), dtype=jnp.float32),
            )

    def explain(self, observation):
        """Return the estimated probability for each final category action.

        This only makes sense if rolls_left > 0.
        """
        cat_logits = self._strategy_network(self._strategy_weights, observation)
        cat_prob = np.exp(cat_logits - cat_logits.max())
        cat_prob /= cat_prob.sum()
        return dict(sorted(enumerate(cat_prob), key=lambda k: k[1], reverse=True))

    def turn(
        self,
        player_scorecard,
        opponent_scorecards=None,
    ):
        """Play a turn.

        This is a generator that yields a dict describing the current state
        after each action.

        Opponent scorecards only have meaning if the objective is "win".
        """
        if self._objective == "win":
            if opponent_scorecards is None or (
                hasattr(opponent_scorecards, "__iter__")
                and len(opponent_scorecards) == 0
            ):
                opponent_scorecards = player_scorecard

            opponent_value = get_opponent_value(
                opponent_scorecards, self._network, self._weights
            )
        else:
            opponent_value = None

        if self._be_greedy:
            roll_lut = get_lut(self._roll_lut_path, self._ruleset)
        else:
            roll_lut = None

        yield from play_turn(
            player_scorecard,
            opponent_value=opponent_value,
            objective=self._objective,
            network=self._network,
            weights=self._weights,
            num_dice=self._ruleset.num_dice,
            use_lut=roll_lut,
        )

    def get_weights(self, strategy=False):
        """Get current network weights."""
        if strategy:
            return self._strategy_weights

        return self._weights

    def set_weights(self, new_weights, strategy=False):
        """Set current network weights."""
        if strategy:
            self._strategy_weights = new_weights
        else:
            self._weights = new_weights

    def clone(self, keep_weights=True):
        """Create a copy of the current agent."""
        yzt = self.__class__(ruleset=self._ruleset.name, objective=self._objective)

        if keep_weights:
            yzt.set_weights(self.get_weights())

        return yzt

    def save(self, path):
        """Save agent to given pickle file."""
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)

        statedict = dict(
            objective=self._objective,
            ruleset=self._ruleset.name,
            weights=self._weights,
            strategy_weights=self._strategy_weights,
        )

        with open(path, "wb") as f:
            pickle.dump(statedict, f)

    def load(self, path):
        """Load agent from given pickle file."""
        with open(path, "rb") as f:
            statedict = pickle.load(f)

        self._objective = statedict["objective"]
        self._ruleset = AVAILABLE_RULESETS[statedict["ruleset"]]
        self._weights = statedict["weights"]
        self._strategy_weights = statedict["strategy_weights"]

    def __repr__(self):
        return f"{self.__class__.__name__}(ruleset={self._ruleset}, objective={self._objective})"
