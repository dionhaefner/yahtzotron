from functools import partial

from loguru import logger
import numpy as np

import jax
import jax.numpy as jnp

from .game import Scorecard


@partial(jax.jit, static_argnums=(1,))
def _convert_rollnet_output(output, num_dice):
    return jnp.unpackbits(output.astype(jnp.uint8))[-num_dice:]
    # return jax.lax.dynamic_slice(unpacked, [-num_dice], [num_dice])


@jax.jit
def _get_roll(current_dice, roll):
    return jnp.sort(jnp.where(current_dice == 0, roll, current_dice))


def _scorecards_to_array(scorecards):
    return jax.tree_map(lambda s: s.to_array(), scorecards)


def turn_fast(player_scorecard, opponent_scorecards, objective, nets, weights, num_dice, num_categories, return_all_actions=False):
    player_scorecard_jax = jnp.array(_scorecards_to_array(player_scorecard))

    if player_scorecard_jax.ndim == 1:
        player_scorecard_jax = jnp.expand_dims(player_scorecard_jax, 0)

    if objective == 'win':
        opponent_scorecard_jax = jnp.array(_scorecards_to_array(opponent_scorecards))
        if opponent_scorecard_jax.ndim == 2:
            opponent_scorecards_jax = jnp.expand_dims(opponent_scorecards_jax, 0)

        strongest_opponent = get_strongest_opponent(
            player_scorecard_jax, opponent_scorecard_jax, nets, weights
        )
    elif objective == 'avg_score':
        # beating someone with equal score means maximizing expected final score
        strongest_opponent = player_scorecard

    rolls = jnp.array(np.random.randint(1, 7, (player_scorecard_jax.shape[0], 3, num_dice)))

    final_roll = get_final_dice(
        player_scorecard_jax, strongest_opponent, rolls, objective, nets, weights, num_dice, return_all_actions
    )

    dice_count = jax.vmap(partial(jnp.bincount, length=7))(final_roll)[:, 1:]

    possible_scorecards = []
    for cat_idx in range(num_categories):
        possible_scorecards_p = []
        for c, p in zip(dice_count, player_scorecard):
            new_scorecard = p.copy()
            try:
                new_scorecard.register_score(c, cat_idx)
            except ValueError:
                # score is already registered
                # TODO: handle this better
                pass

            possible_scorecards_p.append(new_scorecard.to_array())

        possible_scorecards.append(possible_scorecards_p)

    possible_scorecards = jnp.array(possible_scorecards)

    valuenet_in = jnp.concatenate([possible_scorecards, jnp.tile(strongest_opponent, (possible_scorecards.shape[0], 1, 1))], axis=2)
    possible_strength = jax.vmap(nets['value'].apply, in_axes=(None, 0))(weights['value'], valuenet_in)[:, :, 0]
    best_category = jnp.argmax(possible_strength * (1 - player_scorecard_jax[:, :-2].T), axis=0)

    if return_all_actions:
        return dice_count, best_category, recorded_actions

    return dice_count, best_category


@partial(jax.vmap, in_axes=(0, 0, None, None))
def get_strongest_opponent(player_scorecard, opponent_scorecards, nets, weights):
    num_opponents = opponent_scorecards.shape[0]
    valuenet_in = jnp.concatenate([opponent_scorecards, jnp.tile(player_scorecard, (num_opponents, 1))], axis=1)
    opponent_strength = nets['value'].apply(weights['value'], valuenet_in)
    logger.debug('Opponent strengths: {}', opponent_strength)
    strongest_opponent_idx = jnp.argmax(opponent_strength)
    return opponent_scorecards[strongest_opponent_idx]


@partial(jax.vmap, in_axes=(0, 0, 0, None, None, None, None, None))
def get_final_dice(player_scorecard, strongest_opponent, rolls, objective, nets, weights, num_dice, return_all_actions):
    """Play a turn"""
    if return_all_actions:
        recorded_actions = {}

    # - first roll -
    current_dice = jnp.zeros(num_dice, dtype=jnp.uint8)
    roll = _get_roll(current_dice, rolls[0])
    logger.info('Roll 1: {}', roll)

    rollnet_in = jnp.concatenate([roll, player_scorecard, strongest_opponent])
    rollnet_out = nets['roll_1'].apply(weights['roll_1'], rollnet_in)

    action = jnp.argmax(rollnet_out)
    dice_to_keep = _convert_rollnet_output(action, num_dice)

    if return_all_actions:
        recorded_actions['roll_1'] = (rollnet_in, rollnet_out, action)

    current_dice = roll * dice_to_keep
    
    # strategynet_in = jnp.concatenate([
    #     jnp.bincount(current_dice, length=7)[1:],
    #     player_scorecard, strongest_opponent
    # ])
    # strategynet_out = nets['strategy_1'].apply(weights['strategy_1'], strategynet_in)

    # if return_all_actions:
    #     recorded_actions['strategy_1'] = (strategynet_in, strategynet_out)

    # top_3_strategies = jnp.argsort(strategynet_out)[-1:-4:-1]
    # top_3_strategy_names = [categories[i].name for i in top_3_strategies]

    logger.info('Keeping {}', roll)
    # logger.info(' going for {}', top_3_strategies)

    # - second roll -
    roll = _get_roll(current_dice, rolls[1])
    logger.info('Roll 2: {}', roll)

    rollnet_in = jnp.concatenate([roll, player_scorecard, strongest_opponent])
    rollnet_out = nets['roll_2'].apply(weights['roll_2'], rollnet_in)

    action = jnp.argmax(rollnet_out)
    dice_to_keep = _convert_rollnet_output(action, num_dice)

    if return_all_actions:
        recorded_actions['roll_2'] = (rollnet_in, rollnet_out, action)

    current_dice = roll * dice_to_keep

    # strategynet_in = jnp.concatenate([
    #     jnp.bincount(current_dice, length=7)[1:],
    #     player_scorecard, strongest_opponent
    # ])
    # strategynet_out = nets['strategy_2'].apply(weights['strategy_2'], strategynet_in)

    # if return_all_actions:
    #     recorded_actions['strategy_2'] = (strategynet_in,)

    # top_3_strategies = jnp.argsort(strategynet_out)[-1:-4:-1]
    # top_3_strategy_names = [categories[i].name for i in top_3_strategies]

    logger.info('Keeping {}', roll)
    # logger.info(' going for {}', top_3_strategies)

    # - third roll -
    roll = _get_roll(current_dice, rolls[2])
    logger.info('Roll 3: {}', roll)

    if return_all_actions:
        return roll, recorded_actions

    return roll


def print_score(scorecard):
    colwidth = max(len(c.name) for c in scorecard.ruleset_.categories) + 2

    def align(string):
        format_string = f'{{:<{colwidth}}}'
        return format_string.format(string)

    separator_line = ''.join(['=' * colwidth, '+', '=' * 5])

    bonus_total, non_bonus_total = scorecard.score_summary()

    out = [
        '',
        separator_line
    ]

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if not cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = ''.join([align(f' {cat.name}'), '| ', str(score) if filled else ''])
        out.append(line)

    out.append(separator_line)
    out.append(''.join([align(' Bonus'), '| ', str(scorecard.ruleset_.bonus_value(scorecard.scores))]))
    out.append(separator_line)

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = ''.join([align(f' {cat.name}'), '| ', str(score) if filled else ''])
        out.append(line)

    out.append(separator_line)
    out.append(''.join([align(' Total '), '| ', str(bonus_total + non_bonus_total)]))
    out.append(separator_line)
    out.append('')

    return '\n'.join(out)
