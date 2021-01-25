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
        score = self.ruleset_.score(roll, cat_index, self.filled)
        self.scores[cat_index] = score
        self.filled[cat_index] = 1
        return score

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
