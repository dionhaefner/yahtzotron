"""
cli.py

Entry point for CLI.
"""

import os
import sys
import math

import click
from loguru import logger

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
        yzt = train_a2c(yzt, num_epochs=20000, pretraining=True, learning_rate=5e-3)

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
    from yahtzotron.interactive import play_interactive

    if not skip_intro:
        from yahtzotron.eyecandy import play_intro

        play_intro()

    play_interactive(model_path)


@cli.command("evaluate")
@click.argument("AGENTS", nargs=-1, required=True)
@click.option("-n", "--num-rounds", type=click.IntRange(min=0), default=1000)
@click.option(
    "--ruleset", type=click.Choice(list(AVAILABLE_RULESETS.keys())), default="yatzy"
)
@click.option("--deterministic-rolls", is_flag=True, default=False)
def evaluate(agents, num_rounds, ruleset, deterministic_rolls):
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

    scores_per_agent = [[] for _ in agents]
    rank_per_agent = [[] for _ in agents]

    try:
        progress = tqdm.tqdm(range(num_rounds))
        for _ in progress:
            agent_obj = [create_agent(agent_id) for agent_id in agents]
            scorecards = play_tournament(
                agent_obj, deterministic_rolls=deterministic_rolls
            )
            sorted_scores = sorted(
                enumerate([s.total_score() for s in scorecards]),
                key=lambda args: args[1],
                reverse=True,
            )

            for rank, (i, score) in enumerate(sorted_scores, 1):
                scores_per_agent[i].append(score)
                rank_per_agent[i].append(rank)

    except KeyboardInterrupt:
        pass

    for i, agent_id in enumerate(agents):
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
                for k in range(1, len(agents) + 1)
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
