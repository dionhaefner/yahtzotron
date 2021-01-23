import math
import random

from loguru import logger
import tqdm
import numpy as np

import jax
import jax.numpy as jnp

from yahtzotron.game import Scorecard
from yahtzotron.play import turn_fast, print_score


def compute_fitness(scores, objective, tau=0.1):
    if objective == 'win':
        league_size = scores.shape[1]
        league_rankings = np.argsort(scores, axis=1)[:, ::-1]

        rank_per_player = []
        for p in range(league_size):
            player_ranks = []
            for ranking in league_rankings:
                player_rank = ranking.tolist().index(p)
                player_ranks.append(player_rank)
            rank_per_player.append(np.array(player_ranks))

        fitness_per_game = [np.exp(-1 / tau * rank / np.sqrt(league_size)) for rank in rank_per_player]
        return np.mean(fitness_per_game, axis=1)

    if objective == 'avg_score':
        raise NotImplementedError()

    raise ValueError(f'unknown objective {objective}')


def mutate(league, mutate_prob, eps):
    def mutate_one(val):
        mutate_mask = np.random.rand(*val.shape) < mutate_prob
        offset = eps * mutate_mask * np.random.randn(*val.shape)
        return val + offset

    for player in league:
        current_weights = player.get_weights()
        new_weights = jax.tree_multimap(mutate_one, current_weights)
        player.set_weights(new_weights)

    return league


def procreate_asexual(league, fitness, mutate_prob, eps):
    league_size = len(league)
    num_veterans = math.ceil(0.05 * league_size)
    num_new_players = math.ceil(0.1 * league_size)
    num_offspring = league_size - num_new_players - num_veterans

    offspring = [p.clone() for p in random.choices(league, weights=fitness, k=num_offspring)]
    offspring = mutate(offspring, mutate_prob=mutate_prob, eps=eps)

    new_league = (
        [league[p] for p in np.argsort(fitness)[::-1][:num_veterans]]
        + [league[0].clone(keep_weights=False) for _ in range(num_new_players)]
        + offspring
    )
    return new_league


def train_genetic(model, num_epochs, league_size=100, eps=0.1, mutate_prob=0.1, games_per_epoch=100, objective='win'):
    """Train model through self-play"""
    ruleset = model._ruleset
    league = [model.clone(keep_weights=False) for _ in range(league_size)]

    pbar = tqdm.tqdm(total=num_epochs)

    with pbar:
        for i in range(num_epochs):
            scores = [[Scorecard(ruleset) for _ in range(league_size)] for _ in range(games_per_epoch)]

            for t in tqdm.tqdm(range(ruleset.num_rounds)):
                for p in tqdm.tqdm(range(league_size)):
                    my_scores = [s[p] for s in scores]
                    other_scores = [[s for i, s in enumerate(subscores) if i != p] for subscores in scores]
                    roll, best_category = league[p].turn(my_scores, other_scores)

                    for g in range(games_per_epoch):
                        scores[g][p].register_score(roll[g], best_category[g])

            final_scores = []
            for g in range(games_per_epoch):
                game_scores = [s.total_score() for s in scores[g]]
                winner = np.argmax(game_scores)
                logger.warning('Player {} won with a score of {}', winner, scores[g][winner].total_score())
                final_scores.append(game_scores)
            final_scores = np.array(final_scores)

            fitness = compute_fitness(final_scores, objective)
            king_of_the_jungle = np.argmax(fitness)

            logger.warning('League median score: {}', np.median(final_scores))
            logger.warning(' top player median score: {}', np.median(final_scores[:, king_of_the_jungle]))

            best_score = np.unravel_index(np.argmax(final_scores), final_scores.shape)
            logger.warning(' Best scorecard: {}', print_score(scores[best_score[0]][best_score[1]]))

            model.set_weights(league[king_of_the_jungle].get_weights())
            league = procreate_asexual(league, fitness, mutate_prob=mutate_prob, eps=eps)

    return model
