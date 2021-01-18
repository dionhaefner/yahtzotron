def train_backprop(model, num_epochs, league_size=100, eps=1e-2, objective='win'):
    """Train model through self-play"""
    ruleset = model._ruleset

    for i in range(num_epochs):
        league = [model.copy() for _ in range(league_size)]
        scores = [Scorecard(ruleset) for _ in range(league_size)]

        for t in range(ruleset.num_rounds):
            for p in range(league_size):
                roll, best_category = league[p].turn(
                    scores[p],
                    [s for i, s in enumerate(scores) if i != p],
                )
                turn_score = scores[p].register_score(roll, best_category)

        winner = max(range(num_players), key=lambda p: scores[p].total_score())
        logger.warning('Player {} won with a score of {}', winner, scores[winner].total_score())

        # train nets
        for p in range(num_players):
            # roll nets
            winning_bonus = 25 if p == winner else 0
            for n in (1, 2):
                for t in range(ruleset.num_rounds):
                    turn_input = net_actions[p][t][f'roll_{n}'][0]
                    turn_reward = net_actions[p][t][f'roll_{n}'][1][np.newaxis]
                    chosen_action = net_actions[p][t][f'roll_{n}'][2]
                    turn_reward[0, chosen_action] = final_actions[p][t][1] + winning_bonus
                    self._nets[f'roll_{n}'].fit(turn_input, turn_reward, epochs=10, verbose=0)

            # strategy nets
            for n in (1, 2):
                for t in range(ruleset.num_rounds):
                    turn_input = net_actions[p][t][f'strategy_{n}'][0]
                    chosen_action = final_actions[p][t][0]
                    turn_label = to_categorical(chosen_action, num_classes=ruleset.num_categories)[
                        np.newaxis]
                    self._nets[f'strategy_{n}'].fit(
                        turn_input, turn_label, epochs=10, verbose=0)

            # value net
            value_label = np.array([[0]]) if p == winner else np.array([[1]])
            for t in range(ruleset.num_rounds):
                turn_input = net_actions[p][t]['value'][0]
                self._nets['value'].fit(turn_input, value_label, epochs=10, verbose=0)
