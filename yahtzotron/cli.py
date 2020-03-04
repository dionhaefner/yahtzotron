"""
cli.py

Entry point for CLI.
"""

import sys

import click
from loguru import logger

from . import __version__
from .rulesets import AVAILABLE_RULESETS


@click.group('yahtzotron', invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """This is Yahtzotron, the friendly robot that beats you in Yahtzee."""

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


@cli.command('train')
@click.option('-o', '--out', type=click.Path(file_okay=False, writable=True), required=True)
@click.option('--ruleset', type=click.Choice(list(AVAILABLE_RULESETS.keys())), default='yatzy')
@click.option('-n', '--num-epochs', type=click.IntRange(min=0), default=100)
@click.option('-l', '--loglevel', type=click.Choice(['debug', 'info', 'warning']), default='warning')
def train(out, ruleset, num_epochs, loglevel):
    from .model import Yahtzotron

    logger.remove()
    logger.add(sys.stderr, level=loglevel.upper())

    rules = AVAILABLE_RULESETS[ruleset]
    yzt = Yahtzotron(ruleset=rules)
    yzt.train(num_epochs)
    yzt.save(out)


@cli.command('play')
def play():
    pass


@cli.command('evaluate')
def evaluate():
    pass
