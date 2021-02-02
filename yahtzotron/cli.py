"""
cli.py

Entry point for CLI.
"""

import os
import sys

import click
from loguru import logger

from yahtzotron import __version__
from yahtzotron.rulesets import AVAILABLE_RULESETS


@click.group("yahtzotron", invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """This is Yahtzotron, the friendly robot that beats you in Yahtzee."""

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


@cli.command("train")
@click.option(
    "-o", "--out", type=click.Path(file_okay=False, writable=True), required=True
)
@click.option(
    "--ruleset", type=click.Choice(list(AVAILABLE_RULESETS.keys())), default="yatzy"
)
@click.option("-n", "--num-epochs", type=click.IntRange(min=0), default=100)
@click.option("--no-restore", is_flag=True)
@click.option(
    "-v",
    "--loglevel",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="warning",
)
@click.option("--objective", type=click.Choice(["win", "avg_score"]), default="win")
def train(out, ruleset, num_epochs, loglevel, no_restore, objective):
    from yahtzotron.agent import Yahtzotron
    from yahtzotron.training import train_a2c

    logger.remove()
    logger.add(sys.stderr, level=loglevel.upper())

    load_path = None
    if os.path.exists(out) and not no_restore:
        load_path = out

    yzt = Yahtzotron(ruleset=ruleset, objective=objective, load_path=load_path)

    if load_path is None:
        yzt = train_a2c(yzt, num_epochs=10000, pretrain=True)

    yzt = train_a2c(yzt, num_epochs=num_epochs, checkpoint_path=out)

    yzt.save(out)


@cli.command("play")
def play():
    pass


@cli.command("evaluate")
def evaluate():
    pass


if __name__ == "__main__":
    cli()
