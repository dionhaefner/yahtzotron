from collections import deque, defaultdict

import tqdm
import numpy as np
from loguru import logger

import jax
import jax.numpy as jnp
import optax
import rlax

from yahtzotron.game import play_tournament
from yahtzotron.interactive import print_score

REWARD_NORM = 100
WINNING_REWARD = 100

MINIMUM_LOGIT = -1e8


def entropy(logits):
    mask = jnp.isfinite(logits)
    logits = jnp.where(mask, logits, MINIMUM_LOGIT)
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


def cross_entropy(logits, actions):
    mask = jnp.isfinite(logits)
    logprob = jax.nn.log_softmax(jnp.where(mask, logits, MINIMUM_LOGIT))
    labels = jax.nn.one_hot(actions, logits.shape[-1])
    return -jnp.sum(labels * mask * logprob, axis=1)


def compile_loss_function(
    type_, network, td_lambda=0.6, discount=0.99, policy_cost=0.25, entropy_cost=1e-3
):
    def loss(weights, observations, keep_actions, cat_actions, rewards):
        """Actor-critic loss."""
        rolls_left = observations[..., 0]
        keep_logits, cat_logits, values = network(weights, observations)
        values = jnp.append(values, jnp.sum(rewards))

        td_errors = rlax.td_lambda(
            v_tm1=values[:-1],
            r_t=rewards,
            discount_t=jnp.full_like(rewards, discount),
            v_t=values[1:],
            lambda_=jnp.array(td_lambda),
        )
        critic_loss = jnp.mean(td_errors ** 2)

        pertinent_mask = rolls_left == 0
        pertinent_logits = jnp.where(
            pertinent_mask.reshape(-1, 1),
            jnp.pad(cat_logits, ((0, 0), (0, keep_logits.shape[1] - cat_logits.shape[1])), constant_values=-jnp.inf),
            keep_logits
        )
        pertinent_actions = jnp.where(pertinent_mask, cat_actions, keep_actions)

        if type_ == "a2c":
            actor_loss = rlax.policy_gradient_loss(
                logits_t=pertinent_logits,
                a_t=pertinent_actions,
                adv_t=td_errors,
                w_t=jnp.ones(td_errors.shape[0]),
            )
        elif type_ == "supervised":
            actor_loss = jnp.mean(cross_entropy(pertinent_logits, pertinent_actions))

        entropy_loss = -jnp.mean(entropy(pertinent_logits))

        return policy_cost * actor_loss, critic_loss, entropy_cost * entropy_loss

    return jax.jit(loss)


def compile_sgd_step(loss, network, optimizer, max_gradient_norm=0.5):
    def sgd_step(weights, opt_state, observations, keep_actions, cat_actions, rewards):
        """Does a step of SGD over a trajectory."""
        total_loss = lambda *args: sum(loss(*args))
        gradients = jax.grad(total_loss)(weights, observations, keep_actions, cat_actions, rewards)
        gradients = clip_grads(gradients, max_gradient_norm)
        updates, opt_state = optimizer.update(gradients, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    return jax.jit(sgd_step)


def train_a2c(
    base_agent,
    num_epochs,
    checkpoint_path=None,
    players_per_game=4,
    learning_rate=1e-3,
    entropy_cost=1e-3,
    pretraining=False,
):
    """Train advantage actor-critic (A2C) agent through self-play"""
    objective = base_agent._objective

    optimizer = optax.MultiSteps(
        optax.adam(learning_rate), players_per_game, use_grad_mean=False
    )
    opt_state = optimizer.init(base_agent.get_weights())

    running_stats = defaultdict(lambda: deque(maxlen=1000))
    progress = tqdm.tqdm(range(num_epochs), dynamic_ncols=True)

    loss_type = "supervised" if pretraining else "a2c"
    loss_fn = compile_loss_function(
        loss_type, base_agent._network, entropy_cost=entropy_cost
    )
    sgd_step = compile_sgd_step(loss_fn, base_agent._network, optimizer)

    best_score = -float("inf")

    if pretraining:
        greedy_agent = base_agent.clone()
        greedy_agent._default_greedy = True
        agents = [greedy_agent] * players_per_game
    else:
        agents = [base_agent] * players_per_game

    for i in progress:
        scores, trajectories = play_tournament(
            agents,
            record_trajectories=True,
            deterministic_rolls=False if pretraining else True,
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

        weights = base_agent._weights

        for p in range(players_per_game):
            observations, keep_actions, cat_actions, rewards = zip(*trajectories[p])
            assert sum(rewards) == scores[p].total_score()

            observations = np.stack(observations, axis=0)
            keep_actions = np.array(keep_actions, dtype=np.int32)
            cat_actions = np.array(cat_actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32) / REWARD_NORM

            logger.debug(" rewards {}: {}", p, rewards)

            if objective == "win" and p == winner:
                rewards[-1] += WINNING_REWARD / REWARD_NORM

            weights, opt_state = sgd_step(
                weights,
                opt_state,
                observations,
                keep_actions,
                cat_actions,
                rewards,
            )

            loss_components = loss_fn(weights, observations, keep_actions, cat_actions, rewards)
            loss_components = [float(k) for k in loss_components]

            epoch_stats = dict(
                actor_loss=loss_components[0],
                critic_loss=loss_components[1],
                entropy_loss=loss_components[2],
                loss=sum(loss_components),
                score=scores[p].total_score(),
            )
            for key, val in epoch_stats.items():
                buf = running_stats[key]
                if len(buf) == buf.maxlen:
                    buf.popleft()
                buf.append(val)

        base_agent.set_weights(weights)

        if i % 10 == 0:
            avg_score = np.mean(running_stats["score"])
            if avg_score > best_score + 1 and i > running_stats["score"].maxlen:
                best_score = avg_score

                if checkpoint_path is not None:
                    logger.warning(
                        " Saving checkpoint for average score {:.2f}", avg_score
                    )
                    base_agent.save(checkpoint_path)

            progress.set_postfix(
                {key: np.mean(val) for key, val in running_stats.items()}
            )

    return base_agent
