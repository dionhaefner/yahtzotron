import time

import click
import numpy as np

from yahtzotron.agent import Yahtzotron
from yahtzotron.game import Scorecard


def speak(msg, action=None):
    click.echo("> ", nl=False)
    time.sleep(0.2)

    for char in msg:
        click.echo(char, nl=False)
        time.sleep(0.02)

    time.sleep(0.1)

    if action == "confirm":
        return click.confirm("")

    if action == "confirm-abort":
        return click.confirm("", abort=True)

    if action == "prompt":
        return click.prompt("", prompt_suffix=" ")

    click.echo("")


def turn_player(player_scorecard):
    ruleset = player_scorecard.ruleset_
    kept_dice = []

    for rolls_left in (2, 1):
        roll = yield kept_dice

        while True:
            kept_dice = roll.copy()
            keep_values = speak("Which dice do you want to keep?", action="prompt")
            keep_values = [int(k) for k in keep_values if k in "123456"]
            for r in roll:
                if r in keep_values:
                    keep_values.remove(r)
                else:
                    kept_dice.remove(r)

            if len(keep_values) == 0:
                break

            speak("I can't keep more dice than you rolled. Try again.")

        yield dict(rolls_left=rolls_left)

    roll = yield kept_dice

    click.echo("")
    click.echo(
        "\n".join(
            [
                f"{i+1:<3} {cat.name.replace('_', ' ').title()}"
                for i, cat in enumerate(ruleset.categories)
                if not player_scorecard.filled[i]
            ]
        )
    )
    click.echo("")

    while True:
        category_idx = speak("Which category do you choose?", action="prompt")

        try:
            category_idx = int(category_idx) - 1
        except ValueError:
            speak("You need to give me a number. Try again!")
            continue

        if category_idx not in np.where(player_scorecard.filled == 0)[0]:
            speak("That is not an open category. Try again!")
            continue

        break

    yield dict(
        dice_count=np.bincount(roll, minlength=7)[1:],
        category_idx=category_idx,
        rolls_left=0,
    )


def play_interactive(model_path):
    yzt = Yahtzotron(load_path=model_path)

    speak("Greetings.")
    speak("This is Yahtzotron.")

    while True:
        num_players = speak("How many humans am I playing this time?", action="prompt")
        try:
            num_players = int(num_players)
        except ValueError:
            speak("I need a number from you. Try again.")
            continue

        if not 0 <= num_players <= 8:
            speak("I cannot play less than 0 or more than 8 humans. Try again.")
            continue

        break

    auto_rolls = speak("Do you want me to roll the dice for you?", action="confirm")

    def get_roll(current_dice):
        num_dice_to_roll = yzt._ruleset.num_dice - len(current_dice)
        if auto_rolls:
            roll = [*current_dice, *np.random.randint(1, 7, size=num_dice_to_roll)]
            return [int(k) for k in roll]

        if num_dice_to_roll == 0:
            speak("All dice are kept.")
            return current_dice

        while True:
            roll = speak(
                f"Please roll {num_dice_to_roll} dice and tell me the result.",
                action="prompt",
            )
            roll = [int(k) for k in roll if k in "123456"]
            if len(roll) == num_dice_to_roll:
                return [*current_dice, *roll]

            speak(f"I need {num_dice_to_roll} numbers from you. Try again.")

    scorecards = [Scorecard(yzt._ruleset) for _ in range(num_players + 1)]

    categories = yzt._ruleset.categories
    category_names = [cat.name.replace("_", " ").title() for cat in categories]

    def _print_scores():
        align = lambda string: f"{string:<30}"
        score_str = [print_score(s).split("\n") for s in scorecards]
        heading_str = (
            "".join([align(f" Player {n+1}'s score") for n in range(num_players)])
            + " Yahtzotron's score"
        )
        click.echo("")
        click.echo(click.style(" " + heading_str, bold=True))
        for line in zip(*score_str):
            click.echo(" " + "".join([align(s) for s in line]))

    for i_round in range(yzt._ruleset.num_rounds):
        click.echo("")
        click.echo(" " + "=" * 12)
        click.echo(f" Round {i_round+1} / {yzt._ruleset.num_rounds}")
        click.echo(" " + "=" * 12)

        _print_scores()
        time.sleep(2)

        for p, scorecard in enumerate(scorecards):
            is_yahtzotron = p == num_players

            if is_yahtzotron:
                speak("My turn!")
                other_scorecards = scorecards.copy()
                other_scorecards.remove(scorecard)
                turn_iter = yzt.turn(scorecard, other_scorecards)
            else:
                speak(f"Player {p+1}'s turn.")
                turn_iter = turn_player(scorecard)

            for n, kept_dice in enumerate(turn_iter, 1):
                new_roll = sorted(get_roll(kept_dice))
                speak(f"Roll #{n}: {new_roll}.")
                turn_result = turn_iter.send(new_roll)

                if turn_result["rolls_left"] == 0:
                    new_score = scorecard.register_score(
                        turn_result["dice_count"], turn_result["category_idx"]
                    )

                    if is_yahtzotron:
                        time.sleep(2)
                        speak(
                            f"I'll pick the \"{category_names[turn_result['category_idx']]}\" "
                            "category for that."
                        )
                    else:
                        speak(f"OK, I've added {new_score} to your score.")
                    continue

                if is_yahtzotron:
                    time.sleep(1)
                    strategy_prob = yzt.explain(turn_result["observation"])
                    top_strategies = [
                        strat for strat, p in strategy_prob.items() if p > 0.1
                    ][:3]
                    kept_dice = [
                        die
                        for die, keep in zip(new_roll, turn_result["dice_to_keep"])
                        if keep
                    ]
                    strat_str = " or ".join(
                        [category_names[strat] for strat in top_strategies]
                    )
                    speak(
                        f"I think I should go for {strat_str}, so I'm keeping {kept_dice}."
                    )

            click.echo("")
            time.sleep(1)

    click.echo("")
    click.echo(" " + "=" * 12)
    click.echo(" FINAL SCORE")
    click.echo(" " + "=" * 12)

    _print_scores()

    if num_players > 0:
        best_player_score = max(s.total_score() for s in scorecards[:-1])
        yzt_score = scorecards[-1].total_score()

        if best_player_score > yzt_score:
            speak("Looks like the humans won. I bow my head to you in shame.")
        elif best_player_score == yzt_score:
            speak("A tie?! How unsatisfying.")
        else:
            speak("The expected outcome. I won. Good-bye.")


def print_score(scorecard):
    pretty_names = [
        cat.name.replace("_", " ").title() for cat in scorecard.ruleset_.categories
    ]
    colwidth = max(len(pn) for pn in pretty_names) + 2

    def align(string):
        format_string = f"{{:<{colwidth}}}"
        return format_string.format(string)

    separator_line = "".join(["=" * colwidth, "+", "=" * 5])

    out = ["", separator_line]

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if not cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = "".join(
            [align(f" {pretty_names[i]}"), "| ", str(score) if filled else ""]
        )
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

        line = "".join(
            [align(f" {pretty_names[i]}"), "| ", str(score) if filled else ""]
        )
        out.append(line)

    out.append(separator_line)
    out.append("".join([align(" Total "), "| ", str(scorecard.total_score())]))
    out.append(separator_line)
    out.append("")

    return "\n".join(out)
