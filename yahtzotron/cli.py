"""
cli.py

Entry point for CLI.
"""

import os
import sys
import math
import time

import click

from yahtzotron import __version__
from yahtzotron.rulesets import AVAILABLE_RULESETS


@click.group("yahtzotron", invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
@click.option(
    "-v",
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="warning",
)
def cli(ctx, loglevel):
    """This is Yahtzotron, the friendly robot that beats you in Yahtzee."""

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=loglevel.upper())


@cli.command("train")
@click.option(
    "-o", "--out", type=click.Path(dir_okay=False, writable=True), required=True
)
@click.option(
    "--ruleset", type=click.Choice(list(AVAILABLE_RULESETS.keys())), default="yatzy"
)
@click.option("-n", "--num-epochs", type=click.IntRange(min=0), default=100_000)
@click.option("--no-restore", is_flag=True)
@click.option("--objective", type=click.Choice(["win", "avg_score"]), default="win")
def train(out, ruleset, num_epochs, no_restore, objective):
    from yahtzotron.agent import Yahtzotron
    from yahtzotron.training import train_a2c

    load_path = None
    if os.path.exists(out) and not no_restore:
        load_path = out

    yzt = Yahtzotron(ruleset=ruleset, objective=objective, load_path=load_path)

    if load_path is None:
        yzt = train_a2c(yzt, num_epochs=10000, pretraining=True, learning_rate=5e-3)

    stage_1_epochs = round(0.6 * num_epochs)
    yzt = train_a2c(yzt, num_epochs=stage_1_epochs, checkpoint_path=out)
    yzt = train_a2c(
        yzt,
        num_epochs=num_epochs - stage_1_epochs,
        checkpoint_path=out,
        entropy_cost=0,
        learning_rate=1e-4,
    )

    yzt.save(out)


@cli.command("play")
@click.argument("MODEL_PATH")
@click.option("--skip-intro", is_flag=True)
def play(model_path, skip_intro):
    import numpy as np
    from yahtzotron.interactive import play_intro, print_score
    from yahtzotron.agent import Yahtzotron
    from yahtzotron.game import Scorecard

    yzt = Yahtzotron(load_path=model_path)

    if not skip_intro:
        play_intro()

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
            speak(f"The current dice are {current_roll}. Please tell me ")

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
        click.echo(f"\n{' Your score':<30} Yahtzotron's score")
        click.echo("\n".join(f"{s1:<30}{s2}" for s1, s2 in zip(*score_str)))

        roll = get_roll(roll)

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
                kept_dice = [
                    die for die, keep in zip(roll, turn_result["dice_to_keep"]) if keep
                ]
                speak(
                    f"I think I should go for {categories[turn_result['category_idx']].name}, so I'm keeping {kept_dice}."
                )
            else:
                speak(
                    f"I'll pick the \"{categories[turn_result['category_idx']].name}\" category for that."
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


@cli.command("evaluate")
@click.option("-a", "--agent", multiple=True, required=True)
@click.option("-n", "--num-rounds", type=click.IntRange(min=0), default=1000)
@click.option(
    "--ruleset", type=click.Choice(list(AVAILABLE_RULESETS.keys())), default="yatzy"
)
def evaluate(agent, num_rounds, ruleset):
    import tqdm
    import numpy as np

    from yahtzotron.game import play_tournament
    from yahtzotron.agent import Yahtzotron

    def create_agent(agent_id):
        if agent_id == "random":
            return Yahtzotron(ruleset)

        if agent_id == "greedy":
            return Yahtzotron(ruleset, greedy=True)

        return Yahtzotron(ruleset, load_path=agent_id)

    scores_per_agent = [[] for _ in agent]
    rank_per_agent = [[] for _ in agent]

    progress = tqdm.tqdm(range(num_rounds))
    for _ in progress:
        agents = [create_agent(agent_id) for agent_id in agent]
        scorecards = play_tournament(agents, deterministic_rolls=True)
        sorted_scores = sorted(
            enumerate([s.total_score() for s in scorecards]),
            key=lambda args: args[1],
            reverse=True,
        )

        for rank, (i, score) in enumerate(sorted_scores, 1):
            scores_per_agent[i].append(score)
            rank_per_agent[i].append(rank)

    for i, agent_id in enumerate(agent):
        agent_scores = np.asarray(scores_per_agent[i])
        agent_ranks = np.asarray(rank_per_agent[i])
        agent_rank_count = dict(zip(*np.unique(agent_ranks, return_counts=True)))

        summary = []
        summary.append(f"Agent #{i+1} ({agent_id})")
        summary.append("-" * len(summary[-1]))
        summary.extend(
            [
                f' Rank {k} | {"█" * math.ceil(20 * agent_rank_count.get(k, 0) / num_rounds)} '
                f"{agent_rank_count.get(k, 0)}"
                for k in range(1, len(agent) + 1)
            ]
        )
        summary.append(" ---")
        summary.append(
            f" Final score: {np.mean(agent_scores):.1f} ± {np.std(agent_scores):.1f}"
        )
        summary.append("")
        click.echo("\n".join(summary))


if __name__ == "__main__":
    cli()
