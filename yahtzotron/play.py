from functools import partial

import numpy as np
from loguru import logger

import jax
import jax.numpy as jnp
import haiku as hk

from .strategy import reward_idx_to_action

key = hk.PRNGSequence(17)


@jax.jit
def _get_roll(current_dice, key):
    roll = jax.random.randint(key, shape=current_dice.shape, minval=1, maxval=7, dtype=jnp.uint8)
    return jnp.sort(jnp.where(current_dice == 0, roll, current_dice), axis=-1)


def _scorecards_to_array(scorecards):
    return jnp.asarray(jax.tree_map(lambda s: s.to_array(), scorecards))


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, None, 0))
def _apply_valuenet(net, weights, inputs):
    return net.apply(weights, inputs)[:, 0]


@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap, in_axes=(0, None))
def _vector_bincount(x, length):
    return jnp.bincount(x, length=length)


def turn_fast(
    player_scorecard,
    opponent_scorecards,
    objective,
    nets,
    weights,
    num_dice,
    num_categories,
    roll_lut,
    return_all_actions=False,
):
    player_scorecard_jax = _scorecards_to_array(player_scorecard)

    # if player_scorecard_jax.ndim == 1:
    #     player_scorecard_jax = jnp.expand_dims(player_scorecard_jax, 0)

    # if objective == "win":
    #     opponent_scorecard_jax = _scorecards_to_array(opponent_scorecards)
    #     if opponent_scorecard_jax.ndim == 2:
    #         opponent_scorecard_jax = jnp.expand_dims(opponent_scorecard_jax, 0)

    #     strongest_opponent = get_strongest_opponent(
    #         player_scorecard_jax, opponent_scorecard_jax, nets["value"], weights["value"]
    #     )
    # elif objective == "avg_score":
    #     # beating someone with equal score means maximizing expected final score
    strongest_opponent = player_scorecard_jax

    num_games = player_scorecard_jax.shape[0]
    current_dice = jnp.zeros((num_games, num_dice), dtype=jnp.uint8)
    dice_to_keep = jnp.ones_like(current_dice)
    key.reserve(num_games * 6)

    for roll_number in range(3):
        current_dice = _get_roll(current_dice * dice_to_keep, next(key))

        random_keys = jnp.asarray([next(key) for _ in range(num_games)])
        category_idx = get_category_action(
            roll_number,
            current_dice,
            player_scorecard_jax,
            strongest_opponent,
            random_keys,
            nets['strategy'],
            weights['strategy'],
            num_dice,
        )

        if roll_number != 2:
            keep_actions = [np.argmax(roll_lut[(tuple(r), cat_idx)]) for r, cat_idx in zip(current_dice, category_idx)]
            dice_to_keep = jnp.asarray([reward_idx_to_action(a, num_dice) for a in keep_actions])

    logger.info("Picked category {}", category_idx)

    dice_count = _vector_bincount(current_dice, 7)[:, 1:]
    return dice_count, category_idx


@partial(jax.jit, static_argnums=(2,))
@partial(jax.vmap, in_axes=(0, 0, None, None))
def get_strongest_opponent(player_scorecard, opponent_scorecards, net, weights):
    num_opponents = opponent_scorecards.shape[0]
    valuenet_in = jnp.concatenate(
        [opponent_scorecards, jnp.tile(player_scorecard, (num_opponents, 1))], axis=1
    )
    opponent_strength = net.apply(weights, valuenet_in)
    logger.debug("Opponent strengths: {}", opponent_strength)
    strongest_opponent_idx = jnp.argmax(opponent_strength)
    return opponent_scorecards[strongest_opponent_idx]


@partial(jax.jit, static_argnums=(5, 7))
@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, None, None, None))
def get_category_action(
    roll_number,
    current_dice,
    player_scorecard,
    strongest_opponent,
    random_key,
    net,
    weights,
    num_dice,
):
    dice_count = jnp.bincount(current_dice, length=7)[1:]

    strategynet_in = jnp.concatenate([jnp.array([roll_number]), dice_count, player_scorecard, strongest_opponent])
    strategynet_out = net.apply(weights, strategynet_in)
    strategynet_out = jnp.where(player_scorecard[:-2].T == 1, 0, 1e-8 + strategynet_out)

    category_action = jax.random.choice(random_key, len(strategynet_out), p=strategynet_out)
    return category_action


def print_score(scorecard):
    colwidth = max(len(c.name) for c in scorecard.ruleset_.categories) + 2

    def align(string):
        format_string = f"{{:<{colwidth}}}"
        return format_string.format(string)

    separator_line = "".join(["=" * colwidth, "+", "=" * 5])

    bonus_total, non_bonus_total = scorecard.score_summary()

    out = ["", separator_line]

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if not cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = "".join([align(f" {cat.name}"), "| ", str(score) if filled else ""])
        out.append(line)

    out.append(separator_line)
    out.append(
        "".join(
            [
                align(" Bonus"),
                "| ",
                str(scorecard.ruleset_.bonus_value(scorecard.scores)),
            ]
        )
    )
    out.append(separator_line)

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = "".join([align(f" {cat.name}"), "| ", str(score) if filled else ""])
        out.append(line)

    out.append(separator_line)
    out.append("".join([align(" Total "), "| ", str(scorecard.total_score())]))
    out.append(separator_line)
    out.append("")

    return "\n".join(out)
