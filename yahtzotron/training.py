from collections import deque, defaultdict

import tqdm
import numpy as np
from loguru import logger

import jax
import jax.numpy as jnp
import optax
import rlax

from yahtzotron.game import Scorecard
from yahtzotron.play import print_score

REWARD_NORM = 100
WINNING_REWARD = 100


def entropy_loss_fn(logits_t, w_t):
    entropy_per_timestep = entropy_fn(logits_t)
    return -jnp.mean(entropy_per_timestep * w_t)


def entropy_fn(logits):
    mask = jnp.isfinite(logits)
    logits = jnp.where(mask, logits, -1e8)
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
    logprob = jax.nn.log_softmax(jnp.where(mask, logits, -1e8))
    labels = jax.nn.one_hot(actions, logits.shape[-1])
    return -jnp.sum(labels * mask * logprob, axis=1)


def compile_loss_function(
    type_, network, td_lambda=0.9, discount=0.99, entropy_cost=0.01, strategy_cost=0.01, rolls_per_turn=3
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
            actor_loss = (
                rlax.policy_gradient_loss(
                    logits_t=pertinent_logits,
                    a_t=pertinent_actions,
                    adv_t=td_errors,
                    w_t=jnp.ones(td_errors.shape[0]),
                )
            )
        elif type_ == "supervised":
            actor_loss = jnp.mean(cross_entropy(pertinent_logits, pertinent_actions))

        strategy_loss = 0.
        final_actions = cat_actions[rolls_per_turn-1::rolls_per_turn]
        for i in range(0, rolls_per_turn - 1):
            strategy_loss += jnp.mean(cross_entropy(cat_logits[i::rolls_per_turn, :], final_actions))

        entropy_loss = (
            jnp.mean(entropy_loss_fn(keep_logits, jnp.ones(keep_logits.shape[0])))
            + jnp.mean(entropy_loss_fn(cat_logits, jnp.ones(cat_logits.shape[0])))
        )

        return actor_loss, critic_loss, entropy_cost * entropy_loss, strategy_cost * strategy_loss

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


def play_tournament(model, players_per_game, pretrain=False):
    ruleset = model._ruleset
    scores = [Scorecard(ruleset) for _ in range(players_per_game)]
    trajectories = [[] for _ in range(players_per_game)]
    player_values = [-float("inf")] * players_per_game

    for t in range(ruleset.num_rounds):
        for p in range(players_per_game):
            my_score = scores[p]

            player_values[p] = -float("inf")
            strongest_opponent = scores[np.argmax(player_values)]

            turn_iter = model.turn(
                my_score, strongest_opponent, pretrain=pretrain
            )
            for turn_state in turn_iter:
                if turn_state['rolls_left'] == 0:
                    player_values[p] = float(turn_state['value'])
                    reward = scores[p].register_score(turn_state['dice_count'], turn_state['category_idx'])
                else:
                    reward = 0

                trajectories[p].append((turn_state['net_input'], turn_state['keep_action'], turn_state['category_idx'], reward))

    return scores, trajectories, player_values


def train_a2c(
    model,
    num_epochs,
    checkpoint_path=None,
    players_per_game=4,
    learning_rate=1e-3,
    pretrain=False,
):
    """Train model through self-play"""
    objective = model._objective

    optimizer = optax.MultiSteps(
        optax.adam(learning_rate), players_per_game, use_grad_mean=False
    )
    opt_state = optimizer.init(model.get_weights())

    running_stats = defaultdict(lambda: deque(maxlen=1000))
    progress = tqdm.tqdm(range(num_epochs), dynamic_ncols=True)

    loss_type = "supervised" if pretrain else "a2c"
    loss_fn = compile_loss_function(loss_type, model._net)
    sgd_step = compile_sgd_step(loss_fn, model._net, optimizer)

    best_score = -float("inf")

    try:
        for i in progress:
            scores, trajectories, player_values = play_tournament(
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

            for p in range(players_per_game):
                observations, keep_actions, category_actions, rewards = zip(*trajectories[p])
                assert sum(rewards) == scores[p].total_score()

                observations = np.stack(observations, axis=0)
                keep_actions = np.array(keep_actions, dtype=np.int32)
                category_actions = np.array(category_actions, dtype=np.int32)
                rewards = np.array(rewards, dtype=np.float32) / REWARD_NORM

                logger.debug(" rewards {}: {}", p, rewards)

                if objective == "win" and p == winner:
                    rewards[-1] += WINNING_REWARD / REWARD_NORM

                weights, opt_state = sgd_step(
                    weights,
                    opt_state,
                    observations,
                    keep_actions,
                    category_actions,
                    rewards,
                )

                loss_components = loss_fn(weights, observations, keep_actions, category_actions, rewards)
                loss_components = [float(k) for k in loss_components]

                epoch_stats = dict(
                    actor_loss=loss_components[0],
                    critic_loss=loss_components[1],
                    entropy_loss=loss_components[2],
                    strategy_loss=loss_components[3],
                    loss=sum(loss_components),
                    score=scores[p].total_score(),
                )
                for key, val in epoch_stats.items():
                    buf = running_stats[key]
                    if len(buf) == buf.maxlen:
                        buf.popleft()
                    buf.append(val)

            model.set_weights(weights)

            if i % 10 == 0:
                avg_score = np.mean(running_stats["score"])
                if avg_score > best_score + 1 and i > running_stats["score"].maxlen:
                    best_score = avg_score

                    if checkpoint_path is not None:
                        logger.warning(
                            " Saving checkpoint for average score {:.2f}", avg_score
                        )
                        model.save(checkpoint_path)

                progress.set_postfix(
                    {key: np.mean(val) for key, val in running_stats.items()}
                )

    except KeyboardInterrupt:
        pass

    return model
