import time

import click
import numpy as np

from yahtzotron.agent import Yahtzotron
from yahtzotron.game import Scorecard


def play_interactive(model_path):
    yzt = Yahtzotron(load_path=model_path)

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
            return click.prompt("")

        click.echo("")

    speak("This is Yahtzotron.")
    auto_rolls = speak("Do you want me to roll the dice for you?", action="confirm")

    def get_roll(current_roll):
        if auto_rolls:
            return sorted(
                [r if r > 0 else np.random.randint(1, 7) for r in current_roll]
            )

        num_dice_to_roll = yzt._ruleset.num_dice - sum(r > 0 for r in current_roll)
        if num_dice_to_roll == 0:
            speak("All dice are kept.")
            return current_roll

        while True:
            if num_dice_to_roll == yzt._ruleset.num_dice:
                pass
            speak(
                f"The current dice are {current_roll}. Please tell me which ones I should keep."
            )

    player_scorecard = Scorecard(yzt._ruleset)
    yzt_scorecard = Scorecard(yzt._ruleset)

    categories = yzt._ruleset.categories

    for i_round in range(yzt._ruleset.num_rounds):
        click.echo("")
        click.echo("=" * 12)
        click.echo(f"Round {i_round+1} / {yzt._ruleset.num_rounds}")
        click.echo("=" * 12)

        score_str = [
            print_score(s).split("\n") for s in (player_scorecard, yzt_scorecard)
        ]
        click.echo(
            click.style(f"\n{' Your score':<30} Yahtzotron's score", bold=True),
            nl=False,
        )
        click.echo("\n".join(f"{s1:<30}{s2}" for s1, s2 in zip(*score_str)))

        roll = get_roll([0] * yzt._ruleset.num_dice)

        for roll_num in range(1, 3):
            speak(f"Your roll #{roll_num}: {roll}")

            while True:
                keep_values = speak("Which dice do you want to keep?", action="prompt")
                keep_values = [int(k) for k in keep_values if k in "123456"]
                for i, r in enumerate(roll):
                    if r in keep_values:
                        keep_values.remove(r)
                    else:
                        roll[i] = 0
                if len(keep_values) == 0:
                    break

                speak("I can't keep more dice than you rolled. Try again.")

            roll = get_roll(roll)

        speak(f"Your final roll: {roll}")

        click.echo("")
        click.echo(
            "\n".join(
                [
                    f"{i+1:<3} {cat.name}"
                    for i, cat in enumerate(categories)
                    if not player_scorecard.filled[i]
                ]
            )
        )
        click.echo("")

        while True:
            try:
                category_idx = (
                    int(speak("Which category do you choose?", action="prompt")) - 1
                )
                new_score = player_scorecard.register_score(
                    np.bincount(roll, minlength=7)[1:], category_idx
                )
            except Exception:
                speak("Invalid or already filled category. Try again!")
            else:
                break

        speak(f"OK, I've added {new_score} to your score. My turn!")
        click.echo("")

        roll = [0] * yzt._ruleset.num_dice
        turn_iter = yzt.turn(yzt_scorecard, player_scorecard)
        for turn_result in turn_iter:
            if isinstance(turn_result, int):
                turn_result = turn_iter.send(get_roll(roll))

            roll = []
            for die_value, die_count in enumerate(turn_result["dice_count"], 1):
                roll.extend([die_value] * die_count)

            speak(f"I rolled {roll}.")
            time.sleep(0.5)

            if turn_result["rolls_left"] > 0:
                strategy = yzt.explain(turn_result["observation"])
                kept_dice = [
                    die for die, keep in zip(roll, turn_result["dice_to_keep"]) if keep
                ]
                speak(
                    f"I think I should go for {categories[strategy].name}, "
                    f"so I'm keeping {kept_dice}."
                )
            else:
                speak(
                    f"I'll pick the \"{categories[turn_result['category_idx']].name}\" "
                    "category for that."
                )
                yzt_scorecard.register_score(
                    turn_result["dice_count"], turn_result["category_idx"]
                )

            time.sleep(1)

    click.echo("")
    click.echo("=" * 12)
    click.echo("FINAL SCORE")
    click.echo("=" * 12)

    score_str = [print_score(s).split("\n") for s in (player_scorecard, yzt_scorecard)]
    click.echo("\n".join(f"{s1:<30}{s2}" for s1, s2 in zip(*score_str)))

    if player_scorecard.total_score() > yzt_scorecard.total_score():
        speak("Looks like you won. I bow my head to you in shame.")
    elif player_scorecard.total_score() == yzt_scorecard.total_score():
        speak("A tie?! How unsatisfying.")
    else:
        speak("The expected outcome. I won. Good-bye, human.")


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
