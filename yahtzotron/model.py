import os

import numpy as np
from loguru import logger

from tensorflow.keras.layers import (
    Input, Dense, concatenate
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical

from .game import Scorecard, get_roll


def create_roll_net(num_dice, num_categories, learning_rate=0.1):
    """Predicts expected reward for every keep action"""
    action_space = 2 ** num_dice

    inputs = [
        Input(shape=(num_dice,), name='input_roll'),
        Input(shape=(num_categories,), name='input_player_scorecard'),
        Input(shape=(2,), name='input_player_scores'),
        Input(shape=(num_categories,), name='input_opponent_scorecard'),
        Input(shape=(2,), name='input_opponent_scores'),
    ]

    concat_inputs = concatenate(inputs)

    x = Dense(24, activation='relu')(concat_inputs)
    x = Dense(48, activation='relu')(x)
    x = Dense(24, activation='relu')(x)
    outputs = Dense(action_space)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
    return model


def create_strategy_net(num_dice, num_categories, learning_rate=0.1):
    """Predicts category picked after turn"""
    inputs = [
        Input(shape=(6,), name='input_dice_count'),
        Input(shape=(num_categories,), name='input_player_scorecard'),
        Input(shape=(2,), name='input_player_scores'),
        Input(shape=(num_categories,), name='input_opponent_scorecard'),
        Input(shape=(2,), name='input_opponent_scores'),
    ]

    concat_inputs = concatenate(inputs)

    x = Dense(24, activation='relu')(concat_inputs)
    x = Dense(48, activation='relu')(x)
    x = Dense(24, activation='relu')(x)
    x = concatenate([x, inputs[1]])
    outputs = Dense(num_categories, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    return model


def create_value_net(num_categories, learning_rate=0.1):
    """Predicts probability of player 1 winning over player 2"""
    inputs = [
        Input(shape=(num_categories,), name='input_player_scorecard'),
        Input(shape=(2,), name='input_player_scores'),
        Input(shape=(num_categories,), name='input_opponent_scorecard'),
        Input(shape=(2,), name='input_opponent_scores'),
    ]

    concat_inputs = concatenate(inputs)

    x = Dense(24, activation='relu')(concat_inputs)
    x = Dense(48, activation='relu')(x)
    x = Dense(24, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate))
    return model


class Yahtzotron:
    def __init__(self, ruleset, load_path=None, objective='win'):
        self._ruleset = ruleset

        if load_path is None:
            num_dice, num_categories = ruleset.num_dice, ruleset.num_categories
            self._nets = dict(
                roll_1=create_roll_net(num_dice, num_categories, learning_rate=1e-3),
                strategy_1=create_strategy_net(num_dice, num_categories, learning_rate=1e-3),
                roll_2=create_roll_net(num_dice, num_categories, learning_rate=1e-3),
                strategy_2=create_strategy_net(num_dice, num_categories, learning_rate=1e-3),
                value=create_value_net(num_categories, learning_rate=1e-3),
            )
        else:
            self._nets = self.load(load_path)

        possible_objectives = ('win', 'avg_score')
        if objective not in possible_objectives:
            raise ValueError(
                f'Got unexpected objective {objective}, must be one of {possible_objectives}'
            )

        self._objective = objective
        self._epsilon = 0.1
        self._reinit_model = True

    def _convert_rollnet_output(self, output):
        return tuple(int(v) for v in np.binary_repr(output, width=self._ruleset.num_dice))

    @staticmethod
    def _assemble_inputs(input_scorecards, other_scorecards):
        if isinstance(input_scorecards, Scorecard):
            input_scorecards = [input_scorecards]

        if isinstance(other_scorecards, Scorecard):
            other_scorecards = [other_scorecards] * len(input_scorecards)

        return (
            np.array([sc.filled for sc in input_scorecards]),
            np.array([sc.score_summary() for sc in input_scorecards]),
            np.array([sc.filled for sc in other_scorecards]),
            np.array([sc.score_summary() for sc in other_scorecards]),
        )

    def _get_strongest_opponent(player_scorecards, opponent_scorecards):
        if self._objective == 'avg_score':
            # beating someone with equal score means maximizing expected final score
            return player_scorecards

        elif self._objective == 'win':
            # play to win, i.e. compare own score to strongest opponent
            value_inputs = [
                self._assemble_inputs(other_scorecards, player_scorecard)
                for player_scorecard, other_scorecards
                in zip(player_scorecards, opponent_scorecards)
            ]
            opponent_strengths = self._nets['value'].predict(
                value_inputs
            )
            logger.debug('Opponent strengths: {}', opponent_strength)
            strongest_opponent_idx = np.argmax(opponent_strengths, axis=0)
            return opponent_scorecards[strongest_opponent_idx]

    def training_turn(self, player_scorecards, opponent_scorecards, alpha=0.95):
        recorded_actions = {}
        num_games = len(player_scorecards)

        strongest_opponents = self._get_strongest_opponent(player_scorecards, opponent_scorecards)

        # - first roll -
        rolls = np.array([get_roll(num_dice=self._ruleset.num_dice) for _ in range(num_games)])
        rollnet_in = [rolls, *self._assemble_inputs(player_scorecard, strongest_opponent)]
        rollnet_out = self._nets['roll_1'].predict(rollnet_in)

    def turn(self, player_scorecard, opponent_scorecards):
        """Play a turn"""

        # - first roll -
        roll = get_roll(num_dice=self._ruleset.num_dice)
        logger.info('Roll 1: {}', roll)

        rollnet_in = [roll[np.newaxis], *self._assemble_inputs(player_scorecard, strongest_opponent)]
        rollnet_out = self._nets['roll_1'].predict(rollnet_in)[0]

        action = np.argmax(rollnet_out) if np.random.rand() > self._epsilon else np.random.randint(0, len(rollnet_out))
        dice_to_keep = self._convert_rollnet_output(action)

        if return_all_actions:
            recorded_actions['roll_1'] = (rollnet_in, rollnet_out, action)

        kept_dice = [die for die, keep in zip(roll, dice_to_keep) if keep]

        strategynet_in = [
            np.bincount(kept_dice, minlength=7)[np.newaxis, 1:],
            *self._assemble_inputs(player_scorecard, strongest_opponent)
        ]
        strategynet_out = self._nets['strategy_1'].predict(strategynet_in)

        if return_all_actions:
            recorded_actions['strategy_1'] = (strategynet_in, strategynet_out)

        top_3_strategies = np.argsort(strategynet_out[0])[-1:-4:-1]
        top_3_strategy_names = [self._ruleset.categories[i].name for i in top_3_strategies]

        logger.info('Keeping {}', kept_dice)
        logger.info(' going for {}', top_3_strategy_names)

        # - second roll -
        roll = get_roll(kept_dice, num_dice=self._ruleset.num_dice)
        logger.info('Roll 2: {}', roll)

        rollnet_in = [roll[np.newaxis], *self._assemble_inputs(player_scorecard, strongest_opponent)]

        if np.random.rand() > self._epsilon:
            rollnet_out = self._nets['roll_2'].predict(rollnet_in)[0]
            action = np.argmax(rollnet_out)
        else:
            action = np.random.randint(0, 2**self._ruleset.num_dice)

        dice_to_keep = self._convert_rollnet_output(action)

        if return_all_actions:
            recorded_actions['roll_2'] = (rollnet_in, rollnet_out, action)

        kept_dice = [die for die, keep in zip(roll, dice_to_keep) if keep]

        strategynet_in = [
            np.bincount(kept_dice, minlength=7)[np.newaxis, 1:],
            *self._assemble_inputs(player_scorecard, strongest_opponent)
        ]
        strategynet_out = self._nets['strategy_2'].predict(strategynet_in)

        if return_all_actions:
            recorded_actions['strategy_2'] = (strategynet_in,)

        top_3_strategies = np.argsort(strategynet_out[0])[-1:-4:-1]
        top_3_strategy_names = [self._ruleset.categories[i].name for i in top_3_strategies]

        logger.info('Keeping {}', kept_dice)
        logger.info(' going for {}', top_3_strategy_names)

        # - third roll -
        roll = get_roll(kept_dice, num_dice=self._ruleset.num_dice)
        logger.info('Roll 3: {}', roll)

        dice_count = np.bincount(roll, minlength=7)[1:]

        best_category = None
        best_category_strength = -float('inf')
        best_category_score = 0
        best_category_inputs = None

        open_categories = np.where(~player_scorecard.filled)[0]

        possible_scorecards = []

        for cat_idx in open_categories:
            possible_scorecard = player_scorecard.copy()
            possible_scorecard.register_score(dice_count, cat_idx)
            possible_scorecards.append(possible_scorecard)

        valuenet_in = self._assemble_inputs(possible_scorecards, strongest_opponent)

        be_greedy = np.random.rand() < alpha

        if be_greedy:
            logger.debug('Being greedy')
        else:
            logger.debug('Not being greedy')

        possible_strengths = self._nets['value'].predict(valuenet_in)

        for i, cat_idx in enumerate(open_categories):
            cat_strength = possible_strengths[i]
            logger.debug(
                'Category "{}" has strength {}',
                self._ruleset.categories[cat_idx].name,
                cat_strength
            )

            possible_scorecard = player_scorecard.copy()
            cat_score = possible_scorecard.register_score(dice_count, cat_idx)

            if be_greedy:
                is_best_category = cat_score >= best_category_score
            else:
                is_best_category = cat_strength >= best_category_strength

            if is_best_category:
                best_category = cat_idx
                best_category_strength = cat_strength
                best_category_score = cat_score
                best_category_inputs = [v[np.newaxis, i] for v in valuenet_in]

        if return_all_actions:
            recorded_actions['value'] = (best_category_inputs,)

        logger.info(
            'Best category is {} with a score of {}',
            self._ruleset.categories[best_category].name,
            best_category_score
        )

        if return_all_actions:
            return dice_count, best_category, recorded_actions

        return dice_count, best_category

    def pre_train(self, num_samples=100_000):
        """Pre-train value network to go for maximum scores"""
        return
        train_scorecard = np.random.random_integers(0, 1, size=(num_samples, self._ruleset.num_categories))
        train_scores = np.random.random_integers(0, 125, size=(2, num_samples, 2))
        winning_log_odds = (train_scores[0].sum(axis=1) - train_scores[1].sum(axis=1)) / 50
        winning_prob = 1. / (1. + np.exp(-winning_log_odds))
        train_labels = np.random.binomial(1, p=winning_prob, size=(num_samples,))

        self._nets['value'].fit(
            [train_scorecard, train_scores[0], train_scorecard, train_scores[1]],
            train_labels,
            epochs=2
        )

        # train_rolls = np.random.random_integers(1, 6, size=(num_samples, self._ruleset.num_dice))
        # train_dice_counts = np.apply_along_axis(
        #     lambda x: np.bincount(x, minlength=7)[1:],
        #     axis=1, arr=train_rolls
        # )

        # train_rewards = np.array([
        #     [self._ruleset.score(dice_count, cat_idx, scorecard) if not scorecard[cat_idx] else -1 for cat_idx in range(self._ruleset.num_categories)]
        #     for dice_count, scorecard in zip(train_dice_counts, train_scorecard)
        # ])

        # train_best_category = to_categorical(np.argmax(train_rewards, axis=-1))

        # self._nets['strategy_1'].fit(
        #     [train_dice_counts, train_scorecard, train_scores[0], train_scorecard, train_scores[1]],
        #     train_best_category,
        #     epochs=10
        # )

        # self._nets['strategy_2'].fit(
        #     [train_dice_counts, train_scorecard, train_scores[0], train_scorecard, train_scores[1]],
        #     train_best_category,
        #     epochs=10
        # )

    def train(self, num_epochs, num_parallel_games=1000, num_players=2, pre_train=None):
        """Train model through self-play"""
        if pre_train is None:
            pre_train = self._reinit_model

        if pre_train:
            self.pre_train()

        self._reinit_model = False

        for i in range(num_epochs):
            net_actions = [[]] * num_players
            final_actions = [[]] * num_players
            scores = [
                [Scorecard(self._ruleset) for _ in range(num_players)]
                for _ in range(num_parallel_games)
            ]

            for t in range(self._ruleset.num_rounds):
                for p in range(num_players):
                    roll, best_category, turn_actions = self.turn(
                        scores[p],
                        [s for i, s in enumerate(scores) if i != p],
                        return_all_actions=True
                    )
                    turn_score = scores[p].register_score(roll, best_category)
                    net_actions[p].append(turn_actions)
                    final_actions[p].append((best_category, turn_score))

            winner = max(range(num_players), key=lambda p: scores[p].total_score())
            logger.warning('Player {} won with a score of {}', winner, scores[winner].total_score())

            # train nets
            for p in range(num_players):
                final_score = scores[p].total_score()

                # roll nets
                winning_bonus = 150 if p == winner else 0
                for n in (1, 2):
                    for t in range(self._ruleset.num_rounds):
                        turn_input = net_actions[p][t][f'roll_{n}'][0]
                        turn_reward = net_actions[p][t][f'roll_{n}'][1][np.newaxis]
                        chosen_action = net_actions[p][t][f'roll_{n}'][2]
                        turn_reward[0, chosen_action] = final_score + winning_bonus
                        self._nets[f'roll_{n}'].fit(turn_input, turn_reward, epochs=10, verbose=0)

                # strategy nets
                for n in (1, 2):
                    for t in range(self._ruleset.num_rounds):
                        turn_input = net_actions[p][t][f'strategy_{n}'][0]
                        chosen_action = final_actions[p][t][0]
                        turn_label = to_categorical(chosen_action, num_classes=self._ruleset.num_categories)[np.newaxis]
                        self._nets[f'strategy_{n}'].fit(turn_input, turn_label, epochs=10, verbose=0)

                # value net
                value_label = np.array([[1]]) if p == winner else np.array([[0]])
                for t in range(self._ruleset.num_rounds):
                    turn_input = net_actions[p][t]['value'][0]
                    self._nets['value'].fit(turn_input, value_label, epochs=1, verbose=0)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for name, obj in self._nets.items():
            obj.save(os.path.join(path, f'{name}_net.h5'))

    def load(self, path):
        nets = {
            name: load_model(os.path.join(path, f'{name}_net.h5'))
            for name in ('roll_1', 'roll_2', 'value', 'strategy_1', 'strategy_2')
        }
        self._reinit_model = False
        return nets

    def __repr__(self):
        return f'{self.__class__.__name__}(ruleset={self._ruleset}, objective={self._objective})'
