from functools import partial

import numpy as np
from loguru import logger


def turn_fast(
    player_scorecard,
    opponent_scorecard,
    objective,
    net,
    weights,
    num_dice,
    num_categories,
    roll_lut,
    return_all_actions=False,
    greedy=False,
):
    # from pympler import tracker
    # tr = tracker.SummaryTracker()
    if return_all_actions:
        recorded_actions = []

    player_scorecard_arr = player_scorecard.to_array()

    if objective == "win":
        opponent_scorecard_arr = opponent_scorecard.to_array()

    elif objective == "avg_score":
        # beating someone with equal score means maximizing expected final score
        opponent_scorecard_arr = player_scorecard_arr

    current_dice = (0,) * num_dice
    dice_to_keep = (0,) * num_dice

    for roll_number in range(3):
        current_dice = tuple(
            sorted(
                die if keep else np.random.randint(1, 7)
                for die, keep in zip(current_dice, dice_to_keep)
            )
        )
        dice_count = np.bincount(current_dice, minlength=7)[1:]

        cat_out = get_category_action(
            roll_number,
            dice_count,
            player_scorecard_arr,
            opponent_scorecard_arr,
            net,
            weights,
            num_dice,
            return_all_actions,
        )

        if return_all_actions:
            cat_in, category_idx, logits, value = cat_out
        else:
            category_idx, value = cat_out

        if greedy:
            category_idx = get_category_action_greedy(
                roll_number, current_dice, player_scorecard, roll_lut
            )

        category_idx = int(category_idx)

        if roll_number != 2:
            dice_to_keep = max(
                roll_lut["full"][current_dice][category_idx].keys(),
                key=lambda k: roll_lut["full"][current_dice][category_idx][k],
            )

        if return_all_actions:
            recorded_actions.append((roll_number, cat_in, category_idx))
            logger.debug(" Observation: {}", cat_in)
            logger.debug(" Logits: {}", logits)
            logger.debug(" Action: {}", category_idx)
            logger.debug(" Value: {}", value)

    logger.info("Final roll: {} | Picked category: {}", current_dice, category_idx)

    dice_count = np.bincount(current_dice, minlength=7)[1:]

    if return_all_actions:
        return dice_count, category_idx, value, recorded_actions

    return dice_count, category_idx, value


def get_category_action(
    roll_number,
    dice_count,
    player_scorecard,
    opponent_scorecard,
    net,
    weights,
    num_dice,
    return_inputs=False,
):
    strategynet_in = np.concatenate(
        [
            np.array([roll_number]),
            np.array(dice_count),
            player_scorecard,
            opponent_scorecard,
        ]
    )
    policy_logits, value = net(weights, strategynet_in)
    prob = np.exp(policy_logits)
    prob /= prob.sum()
    category_action = np.random.choice(policy_logits.shape[0], p=prob)

    if return_inputs:
        return strategynet_in, category_action, policy_logits, value

    return category_action, value


def get_category_action_greedy(roll_number, current_dice, player_scorecard, roll_lut):
    # greedily pick action with highest expected reward advantage
    # this is not optimal play but should be a good baseline
    num_dice = player_scorecard.ruleset_.num_dice
    num_categories = player_scorecard.ruleset_.num_categories

    if roll_number == 2:
        # there is no keep action, so only keeping all dice counts
        best_payoff = lambda lut, cat: lut[cat][(1,) * num_dice]
        marginal_lut = roll_lut["marginal-0"]
    else:
        best_payoff = lambda lut, cat: max(lut[cat].values())
        marginal_lut = roll_lut["marginal-1"]

    expected_payoff = [
        (
            best_payoff(roll_lut["full"][current_dice], c)
            if player_scorecard.filled[c] == 0
            else -float("inf")
        )
        - marginal_lut[c]
        for c in range(num_categories)
    ]
    category_idx = np.argmax(expected_payoff)
    return category_idx


def print_score(scorecard):
    colwidth = max(len(c.name) for c in scorecard.ruleset_.categories) + 2

    def align(string):
        format_string = f"{{:<{colwidth}}}"
        return format_string.format(string)

    separator_line = "".join(["=" * colwidth, "+", "=" * 5])

    bonus_total, non_bonus_total = scorecard.score_summary()

    out = ["", separator_line]

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if not cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = "".join([align(f" {cat.name}"), "| ", str(score) if filled else ""])
        out.append(line)

    out.append(separator_line)
    out.append(
        "".join(
            [
                align(" Bonus"),
                "| ",
                str(scorecard.ruleset_.bonus_value(scorecard.scores)),
            ]
        )
    )
    out.append(separator_line)

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = "".join([align(f" {cat.name}"), "| ", str(score) if filled else ""])
        out.append(line)

    out.append(separator_line)
    out.append("".join([align(" Total "), "| ", str(scorecard.total_score())]))
    out.append(separator_line)
    out.append("")

    return "\n".join(out)
