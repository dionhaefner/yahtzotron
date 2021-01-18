from loguru import logger
import tqdm

from yahtzotron.game import Scorecard


def compute_fitness(league, objective):
    pass


def procreate_asexual(league, fitness):
    pass


def train_genetic(model, num_epochs, league_size=100, eps=1e-2, objective='win'):
    """Train model through self-play"""
    ruleset = model._ruleset
    league = [model.copy() for _ in range(league_size)]

    pbar = tqdm.tqdm(total=num_epochs)

    with pbar:
        for i in range(num_epochs):
            scores = [Scorecard(ruleset) for _ in range(league_size)]

            for t in range(ruleset.num_rounds):
                for p in range(league_size):
                    other_scores = [s for i, s in enumerate(scores) if i != p]
                    roll, best_category = league[p].turn(scores[p], other_scores,)
                    scores[p].register_score(roll, best_category)

            winner = max(range(league_size), key=lambda p: scores[p].total_score())
            logger.warning('Player {} won with a score of {}', winner, scores[winner].total_score())

            model.__setstate__(league[winner].__getstate__)

            fitness = compute_fitness(league, objective)
            league = procreate_asexual(league, fitness)

    return model
