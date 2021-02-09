from copy import deepcopy
import numpy as np


class Scorecard:
    def __init__(self, ruleset, scores=None):
        self.scores = np.zeros(ruleset.num_categories, dtype="int")
        self.filled = np.zeros(ruleset.num_categories, dtype="bool")

        if scores is not None:
            assert len(scores) == ruleset.num_categories
            for i in range(ruleset.num_categories):
                if scores[i] is not None:
                    self.scores[i] = scores[i]
                    self.filled[i] = 1

        self.ruleset_ = ruleset

    def copy(self):
        new_card = Scorecard(self.ruleset_)
        new_card.scores[...] = self.scores
        new_card.filled[...] = self.filled
        return new_card

    def register_score(self, roll, cat_index):
        roll = np.array(roll)
        total_score_old = self.total_score()
        score = self.ruleset_.score(roll, cat_index, self.filled)
        self.scores[cat_index] = score
        self.filled[cat_index] = 1
        return self.total_score() - total_score_old

    def score_summary(self):
        return np.array(self.ruleset_.score_summary(self.scores))

    def total_score(self):
        return self.ruleset_.total_score(self.scores)

    def to_array(self):
        return np.concatenate([self.filled, self.score_summary()])

    def __repr__(self):
        score_str = ", ".join(
            str(score) if filled else "None"
            for score, filled in zip(self.scores, self.filled)
        )
        return (
            f"{self.__class__.__name__}(ruleset={self.ruleset_}, scores=[{score_str}])"
        )


def play_tournament(agents, deterministic_rolls=False, record_trajectories=False):
    if not hasattr(agents, "__iter__"):
        agents = [agents]

    num_players = len(agents)

    ruleset = agents[0]._ruleset
    for a in agents:
        if a._ruleset != ruleset:
            raise ValueError("Only agents sharing the same ruleset can play each other")

    num_dice = ruleset.num_dice

    scores = [Scorecard(ruleset) for _ in range(num_players)]

    if record_trajectories:
        trajectories = [[] for _ in range(num_players)]

    for t in range(ruleset.num_rounds):
        frozen_scores = deepcopy(scores)

        if deterministic_rolls:
            rng_seed = np.random.randint(int(1e10))
        else:
            rng_seed = None

        for p in range(num_players):
            rng = np.random.default_rng(rng_seed)

            my_score = frozen_scores[p]
            other_scores = [frozen_scores[i] for i in range(num_players) if i != p]

            turn_iter = agents[p].turn(my_score, other_scores)
            for turn_state in turn_iter:
                if isinstance(turn_state, int):
                    dice_roll = rng.integers(1, 7, size=num_dice)[:turn_state]
                    turn_state = turn_iter.send(dice_roll)

                if turn_state["rolls_left"] == 0:
                    reward = scores[p].register_score(
                        turn_state["dice_count"], turn_state["category_idx"]
                    )
                else:
                    reward = 0

                if record_trajectories:
                    trajectories[p].append(
                        (turn_state["net_input"], turn_state["keep_action"], turn_state["category_idx"], reward)
                    )

    if record_trajectories:
        return scores, trajectories

    return scores
