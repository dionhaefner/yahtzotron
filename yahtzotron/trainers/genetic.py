import random
import multiprocessing

from loguru import logger
import tqdm
import numpy as np

import jax

from yahtzotron.game import Scorecard
from yahtzotron.play import print_score


def compute_fitness(scores, objective, tau=0.1):
    if objective == "win":
        league_size = scores.shape[1]
        league_rankings = np.argsort(scores, axis=1)[:, ::-1]

        rank_per_player = []
        for p in range(league_size):
            player_ranks = []
            for ranking in league_rankings:
                player_rank = 1 + ranking.tolist().index(p)
                player_ranks.append(player_rank)
            rank_per_player.append(player_ranks)
        rank_per_player = np.asarray(rank_per_player)

        fitness_per_game = (
            1 / rank_per_player ** 2
        )
        return np.mean(fitness_per_game, axis=1)

    if objective == "avg_score":
        return 0.01 * np.mean(scores, axis=0) ** 2

    raise ValueError(f"unknown objective {objective}")


def mutate(league, eps, mutate_prob=0.01):
    def mutate_one(val):
        return val + eps * np.random.randn(*val.shape) * (
            np.random.rand(*val.shape) < mutate_prob
        )

    for player in league:
        current_weights = player.get_weights()
        new_weights = jax.tree_multimap(mutate_one, current_weights)
        player.set_weights(new_weights)

    return league


def procreate_asexual(league, fitness, eps):
    league_size = len(league)
    num_veterans = max(int(0.2 * league_size), 1)
    num_new_players = 0 #max(int(0.1 * league_size), 1)
    num_offspring = league_size - num_new_players - num_veterans

    offspring_idx = random.choices(range(league_size), weights=fitness, k=num_offspring)
    logger.warning(
        " offspring: {}", dict(zip(*np.unique(offspring_idx, return_counts=True)))
    )
    offspring = [league[p].clone() for p in offspring_idx]
    offspring = mutate(offspring, eps=eps)

    new_league = (
        [league[p] for p in np.argsort(fitness)[-num_veterans:]]
        + [league[0].clone(keep_weights=False) for _ in range(num_new_players)]
        + offspring
    )
    return new_league


def procreate_sexual(league, fitness, eps):
    """When I get that feeling..."""
    league_size = len(league)
    num_veterans = max(int(0.1 * league_size), 1)
    num_new_players = max(int(0.1 * league_size), 1)
    num_offspring = league_size - num_new_players - num_veterans

    p = fitness / fitness.sum()
    couples = np.sort(
        [
            np.random.choice(league_size, size=2, replace=False, p=p)
            for _ in range(num_offspring)
        ],
        axis=1,
    )
    # logger.warning(' offspring: {}', dict(zip(*np.unique(couples, axis=0, return_counts=True))))
    offspring = [league[0].clone(keep_weights=False) for _ in range(num_offspring)]

    for (p1, p2), o in zip(couples, offspring):
        new_weights = jax.tree_multimap(
            lambda x1, x2: 0.5 * (x1 + x2),
            dict(league[p1].get_weights()),
            dict(league[p2].get_weights()),
        )
        o.set_weights(new_weights)

    offspring = mutate(offspring, eps=eps)

    new_league = (
        [league[p] for p in np.argsort(fitness)[-num_veterans:]]
        + [league[0].clone(keep_weights=False) for _ in range(num_new_players)]
        + offspring
    )

    return new_league


def plot_state(pipe):
    import matplotlib.pyplot as plt

    p_output, p_input = pipe
    p_input.close()

    fig = plt.figure()
    plt.xlabel("Generation")
    plt.ylabel("Score")

    plt.show(block=False)

    i = 1

    try:
        while True:
            if p_output.poll(0.01):
                msg = p_output.recv()
                if msg is None:
                    break

                plt.boxplot(msg.flatten(), positions=[i])
                plt.xlim(max(0, i - 40), i + 1)
                plt.ylim(0, 300)
                i += 1

            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.1)
    except KeyboardInterrupt:
        pass

    plt.close(fig)


def train_genetic(model, num_epochs, league_size=10, eps=0.1, games_per_epoch=10, restart=False):
    """Train model through self-play"""
    ruleset = model._ruleset
    objective = model._objective

    if restart:
        league = [model.clone() for _ in range(league_size)]
        mutate(league[1:], eps)
    else:
        league = [model.clone(keep_weights=False) for _ in range(league_size)]

    p_output, p_input = multiprocessing.Pipe()
    plot_p = multiprocessing.Process(target=plot_state, args=((p_output, p_input),))
    plot_p.daemon = True
    plot_p.start()
    p_output.close()

    try:
        for i in tqdm.tqdm(range(num_epochs), position=0):
            scores = [
                [Scorecard(ruleset) for _ in range(league_size)]
                for _ in range(games_per_epoch)
            ]

            # randomize turn order every epoch
            turn_order = list(range(league_size))
            random.shuffle(turn_order)

            for t in tqdm.tqdm(range(ruleset.num_rounds), position=1, leave=False):
                for p in tqdm.tqdm(turn_order, position=2, leave=False):
                    my_scores = [s[p] for s in scores]
                    other_scores = [
                        [s for i, s in enumerate(subscores) if i != p]
                        for subscores in scores
                    ]
                    roll, best_category = league[p].turn(my_scores, other_scores)

                    for g in range(games_per_epoch):
                        scores[g][p].register_score(roll[g], best_category[g])

            final_scores = []
            for g in range(games_per_epoch):
                game_scores = [s.total_score() for s in scores[g]]
                winner = np.argmax(game_scores)
                logger.warning(
                    "Player {} won with a score of {}",
                    winner,
                    scores[g][winner].total_score(),
                )
                final_scores.append(game_scores)
            final_scores = np.array(final_scores)

            p_input.send(final_scores)

            fitness = compute_fitness(final_scores, objective)
            king_of_the_jungle = np.argmax(fitness)

            logger.warning("League median score: {}", np.median(final_scores))
            logger.warning(
                " top player ({}) median score: {}",
                king_of_the_jungle,
                np.median(final_scores[:, king_of_the_jungle]),
            )

            best_score = np.unravel_index(np.argmax(final_scores), final_scores.shape)
            logger.warning(
                " Best scorecard:\n{}",
                print_score(scores[best_score[0]][best_score[1]]),
            )

            model.set_weights(league[king_of_the_jungle].get_weights())
            league = procreate_asexual(league, fitness, eps=eps)
            # league = procreate_sexual(league, fitness, eps=eps / 2)

    except KeyboardInterrupt:
        return model

    finally:
        p_input.send(None)
        p_input.close()
        plot_p.join()

    return model
