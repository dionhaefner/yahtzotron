from functools import partial
from collections import deque

import tqdm
import numpy as np
from loguru import logger

import jax
import jax.numpy as jnp
import optax
import rlax

from yahtzotron.game import Scorecard
from yahtzotron.play import print_score

WINNING_REWARD = 1


def entropy_loss_fn(logits_t, w_t):
    entropy_per_timestep = entropy_fn(logits_t)
    return -jnp.mean(entropy_per_timestep * w_t)


def entropy_fn(logits):
    mask = jnp.isfinite(logits)
    logits = jnp.where(mask, logits, -1e5)
    probs = jax.nn.softmax(logits)
    logprobs = jax.nn.log_softmax(logits)
    return -jnp.sum(mask * probs * logprobs, axis=-1)


def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = l2_norm(grad_tree)
    normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
    return jax.tree_util.tree_map(normalize, grad_tree)


@partial(jax.jit, static_argnums=(0,))
def a2c_loss(network, weights, observations, actions, rewards):
    """Actor-critic loss."""
    td_lambda = 0.9
    discount = 0.99
    entropy_cost = 0.01

    logits, values = network(weights, observations)
    values = jnp.append(values, jnp.sum(rewards))

    td_errors = rlax.td_lambda(
        v_tm1=values[:-1],
        r_t=rewards,
        discount_t=jnp.full_like(rewards, discount),
        v_t=values[1:],
        lambda_=jnp.array(td_lambda),
    )
    critic_loss = jnp.mean(td_errors ** 2)

    actor_loss = rlax.policy_gradient_loss(
        logits_t=logits,
        a_t=actions,
        adv_t=td_errors,
        w_t=jnp.ones(td_errors.shape[0]),
    )

    entropy_loss = jnp.mean(entropy_loss_fn(logits, jnp.ones(logits.shape[0])))
    return actor_loss, critic_loss, entropy_cost * entropy_loss


@partial(jax.jit, static_argnums=(0,))
def supervised_loss(network, weights, observations, actions, rewards):
    td_lambda = 0.9
    discount = 0.99
    entropy_cost = 0.0

    logits, values = network(weights, observations)
    values = jnp.append(values, jnp.sum(rewards))

    td_errors = rlax.td_lambda(
        v_tm1=values[:-1],
        r_t=rewards,
        discount_t=jnp.full_like(rewards, discount),
        v_t=values[1:],
        lambda_=jnp.array(td_lambda),
    )
    critic_loss = jnp.mean(td_errors ** 2)

    # use cross-entropy loss for actor
    mask = jnp.isfinite(logits)
    logprob = jax.nn.log_softmax(jnp.where(mask, logits, -1e5))
    labels = jax.nn.one_hot(actions, logits.shape[-1])
    actor_loss = -jnp.sum(labels * mask * logprob) / labels.shape[0]

    entropy_loss = jnp.mean(entropy_loss_fn(logits, jnp.ones(logits.shape[0])))
    return actor_loss, critic_loss, entropy_cost * entropy_loss


@partial(jax.jit, static_argnums=(0, 1, 3))
def sgd_step(
    loss, network, weights, optimizer, opt_state, observations, actions, rewards
):
    """Does a step of SGD over a trajectory."""
    max_gradient_norm = 0.5
    total_loss = lambda *args: sum(loss(*args))
    gradients = jax.grad(total_loss, 1)(
        network, weights, observations, actions, rewards
    )
    gradients = clip_grads(gradients, max_gradient_norm)
    updates, opt_state = optimizer.update(gradients, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state


def play_game(model, players_per_game, pretrain=False):
    ruleset = model._ruleset
    scores = [Scorecard(ruleset) for _ in range(players_per_game)]
    trajectories = [[] for _ in range(players_per_game)]
    player_values = [-float("inf")] * players_per_game

    for t in range(ruleset.num_rounds):
        for p in range(players_per_game):
            my_score = scores[p]

            player_values[p] = -float("inf")
            strongest_opponent = scores[np.argmax(player_values)]

            roll, best_category, value, recorded_actions = model.turn(
                my_score, strongest_opponent, return_all_actions=True, pretrain=pretrain
            )

            player_values[p] = float(value)
            new_score = scores[p].register_score(roll, best_category)

            for nroll, inputs, action in recorded_actions:
                reward = new_score if nroll == 2 else 0
                trajectories[p].append((inputs, action, reward))

    return scores, trajectories, player_values


def train_a2c(
    model, num_epochs, players_per_game=4, learning_rate=1e-3, pretrain=False
):
    """Train model through self-play"""
    objective = model._objective

    optimizer = optax.MultiSteps(
        optax.adam(learning_rate), players_per_game, use_grad_mean=False
    )
    opt_state = optimizer.init(model.get_weights())

    running_stats = {
        k: deque(maxlen=1000)
        for k in ("score", "loss", "actor_loss", "critic_loss", "entropy_loss")
    }
    progress = tqdm.tqdm(range(num_epochs))

    try:
        for i in progress:
            scores, trajectories, player_values = play_game(
                model, players_per_game, pretrain
            )

            final_scores = [s.total_score() for s in scores]
            winner = np.argmax(final_scores)
            logger.info(
                "Player {} won with a score of {} (median {})",
                winner,
                final_scores[winner],
                np.median(final_scores),
            )
            logger.info(
                " Winning scorecard:\n{}",
                print_score(scores[winner]),
            )

            weights = model._weights
            loss_fn = supervised_loss if pretrain else a2c_loss

            for p in range(players_per_game):
                observations, actions, rewards = zip(*trajectories[p])
                assert sum(rewards) == scores[p].total_score()

                observations = np.stack(observations, axis=0)
                actions = np.array(actions, dtype=np.int32)
                rewards = np.array(rewards, dtype=np.float32) / 100

                logger.debug(" rewards {}: {}", p, rewards)

                if objective == "win" and p == winner:
                    rewards[-1] += WINNING_REWARD

                loss_components = loss_fn(
                    model._net, weights, observations, actions, rewards
                )

                epoch_stats = dict(
                    actor_loss=loss_components[0],
                    critic_loss=loss_components[1],
                    entropy_loss=loss_components[2],
                    loss=sum(loss_components),
                    score=scores[p].total_score(),
                )
                for key, buf in running_stats.items():
                    if len(buf) == buf.maxlen:
                        buf.popleft()
                    buf.append(epoch_stats[key])

                weights, opt_state = sgd_step(
                    loss_fn,
                    model._net,
                    weights,
                    optimizer,
                    opt_state,
                    observations,
                    actions,
                    rewards,
                )

            model.set_weights(weights)

            if i % 10 == 0:
                progress.set_postfix(
                    {key: np.median(val) for key, val in running_stats.items()}
                )

            logger.info(" loss: {:.2e}", sum(loss_components))

    except KeyboardInterrupt:
        pass

    return model
