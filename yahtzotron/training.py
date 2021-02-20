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

MINIMUM_LOGIT = jnp.finfo(jnp.float32).min


def entropy(logits):
    probs = jax.nn.softmax(logits)
    logprobs = jax.nn.log_softmax(logits)
    return -jnp.sum(probs * logprobs, axis=-1)


def cross_entropy(logits, actions):
    logprob = jax.nn.log_softmax(logits)
    labels = jax.nn.one_hot(actions, logits.shape[-1])
    return -jnp.sum(labels * logprob, axis=1)


def compile_loss_function(type_, network):
    def loss(
        weights,
        observations,
        actions,
        rewards,
        td_lambda=0.2,
        discount=0.99,
        policy_cost=0.25,
        entropy_cost=1e-3,
    ):
        """Actor-critic loss."""
        logits, values = network(weights, observations)
        values = jnp.append(values, jnp.sum(rewards))

        # replace -inf values by tiny finite value
        logits = jnp.maximum(logits, MINIMUM_LOGIT)

        td_errors = rlax.td_lambda(
            v_tm1=values[:-1],
            r_t=rewards,
            discount_t=jnp.full_like(rewards, discount),
            v_t=values[1:],
            lambda_=td_lambda,
        )
        critic_loss = jnp.mean(td_errors ** 2)

        if type_ == "a2c":
            actor_loss = rlax.policy_gradient_loss(
                logits_t=logits,
                a_t=actions,
                adv_t=td_errors,
                w_t=jnp.ones(td_errors.shape[0]),
            )
        elif type_ == "supervised":
            actor_loss = jnp.mean(cross_entropy(logits, actions))

        entropy_loss = -jnp.mean(entropy(logits))

        return policy_cost * actor_loss, critic_loss, entropy_cost * entropy_loss

    return jax.jit(loss)


def compile_sgd_step(loss_func, optimizer):
    def sgd_step(weights, opt_state, observations, actions, rewards, **loss_kwargs):
        """Does a step of SGD over a trajectory."""
        total_loss = lambda *args: sum(loss_func(*args, **loss_kwargs))
        gradients = jax.grad(total_loss)(weights, observations, actions, rewards)
        updates, opt_state = optimizer.update(gradients, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    return jax.jit(sgd_step)


def get_default_schedules(pretraining=False):
    if pretraining:
        return dict(
            learning_rate=optax.constant_schedule(5e-3),
            entropy=optax.constant_schedule(1e-3),
            td_lambda=optax.constant_schedule(0.2),
        )

    return dict(
        learning_rate=optax.exponential_decay(1e-3, 60_000, decay_rate=0.2),
        entropy=optax.exponential_decay(
            1e-3, 10_000, decay_rate=0.1, transition_begin=60_000
        ),
        td_lambda=optax.polynomial_schedule(0.2, 0.8, power=1, transition_steps=60_000),
    )


def train_a2c(
    base_agent,
    num_epochs,
    checkpoint_path=None,
    players_per_game=4,
    lr_schedule=None,
    entropy_schedule=None,
    td_lambda_schedule=None,
    pretraining=False,
):
    """Train advantage actor-critic (A2C) agent through self-play"""
    objective = base_agent._objective

    default_schedules = get_default_schedules(pretraining=pretraining)

    if lr_schedule is None:
        lr_schedule = default_schedules["learning_rate"]

    if entropy_schedule is None:
        entropy_schedule = default_schedules["entropy"]

    if td_lambda_schedule is None:
        td_lambda_schedule = default_schedules["td_lambda"]

    optimizer = optax.MultiSteps(
        optax.chain(optax.adam(learning_rate=1), optax.scale_by_schedule(lr_schedule)),
        players_per_game,
    )
    opt_state = optimizer.init(base_agent.get_weights())

    running_stats = defaultdict(lambda: deque(maxlen=1000))
    progress = tqdm.tqdm(range(num_epochs), dynamic_ncols=True)

    loss_type = "supervised" if pretraining else "a2c"
    loss_fn = compile_loss_function(loss_type, base_agent._network)
    sgd_step = compile_sgd_step(loss_fn, optimizer)

    best_score = -float("inf")

    if pretraining:
        greedy_agent = base_agent.clone()
        greedy_agent._be_greedy = True
        agents = [greedy_agent] * players_per_game
    else:
        agents = [base_agent] * players_per_game

    for i in progress:
        scores, trajectories = play_tournament(agents, record_trajectories=True)

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
        loss_kwargs = dict(
            entropy_cost=entropy_schedule(i), td_lambda=td_lambda_schedule(i)
        )

        for p in range(players_per_game):
            observations, actions, rewards = zip(*trajectories[p])
            assert sum(rewards) == scores[p].total_score()

            observations = np.stack(observations, axis=0)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32) / REWARD_NORM

            logger.debug(" observations {}: {}", p, observations)
            logger.debug(" actions {}: {}", p, actions)
            logger.debug(" rewards {}: {}", p, rewards)

            if objective == "win" and p == winner:
                rewards[-1] += WINNING_REWARD / REWARD_NORM

            weights, opt_state = sgd_step(
                weights, opt_state, observations, actions, rewards, **loss_kwargs
            )

            loss_components = loss_fn(
                weights, observations, actions, rewards, **loss_kwargs
            )
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

        if pretraining:
            greedy_agent.set_weights(weights)

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


def train_strategy(agent, num_epochs, players_per_game=4, learning_rate=1e-3):
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(agent.get_weights(strategy=True))

    num_rolls = 3

    @jax.jit
    def loss_fn(weights, observation, action, *args):
        loss = jnp.zeros(1)

        final_actions = action[num_rolls - 1 :: num_rolls]

        for i_roll in range(num_rolls - 1):
            logits = agent._strategy_network(weights, observation[i_roll::num_rolls])
            loss = loss + jnp.mean(cross_entropy(logits, final_actions))

        return loss

    sgd_step = compile_sgd_step(loss_fn, optimizer)
    running_loss = deque(maxlen=1000)

    progress = tqdm.tqdm(range(num_epochs), dynamic_ncols=True)

    for i in progress:
        _, trajectories = play_tournament(
            [agent] * players_per_game, record_trajectories=True
        )

        weights = agent.get_weights(strategy=True)

        for p in range(players_per_game):
            observations, actions, _ = zip(*trajectories[p])

            observations = np.stack(observations, axis=0)
            actions = np.array(actions, dtype=np.int32)

            rolls_left = observations[..., 0]
            for k in range(num_rolls):
                assert np.all(rolls_left[k::num_rolls] == num_rolls - k - 1)

            logger.debug(" observations {}: {}", p, observations)
            logger.debug(" actions {}: {}", p, actions)

            weights, opt_state = sgd_step(
                weights,
                opt_state,
                observations,
                actions,
                None,
            )

            loss = float(loss_fn(weights, observations, actions))

            if len(running_loss) == running_loss.maxlen:
                running_loss.popleft()

            running_loss.append(loss)

        agent.set_weights(weights, strategy=True)

        if i % 10 == 0:
            progress.set_postfix(dict(loss=np.mean(running_loss)))

    return agent
