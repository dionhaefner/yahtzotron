# Yahtzotron

> State your prime directive! - "... to ... roll ..." 🤖 🎲

Yahtzotron is a bot for [Yahtzee](https://en.wikipedia.org/wiki/Yahtzee) and [Yatzy](https://en.wikipedia.org/wiki/Yatzy), trained via advantage actor-critic (A2C) through self-play. Yahtzotron is implemented through the JAX library ecosystem ([JAX](https://github.com/google/jax) + [Haiku](https://github.com/deepmind/dm-haiku) + [optax](https://github.com/deepmind/optax) + [rlax](https://github.com/deepmind/rlax)).

Yahtzee is a game of chance played with 5 dice and involves making strategic decisions based on the outcome of your rolls early in the game. This makes for a surprisingly challenging task for reinforcement learning.

The pre-trained agents are close to perfect play (average scores are around 240 for both Yahtzee and Yatzy, just 5-10 points below perfect play).

[Read my blog post about the making of Yahtzotron here.](https://dionhaefner.github.io/2021/04/yahtzotron-learning-to-play-yahtzee-with-advantage-actor-critic/)

## Usage

Just clone the repository and run

```bash
$ pip install .
```

Then, you can use the Yahtzotron command-line interface:

```
$ yahtzotron --help
Usage: yahtzotron [OPTIONS] COMMAND [ARGS]...

  This is Yahtzotron, the friendly robot that beats you in Yahtzee.

Options:
  --version                       Show the version and exit.
  -v, --loglevel [debug|info|warning|error]
  --help                          Show this message and exit.

Commands:
  evaluate  Evaluate performance of trained agents.
  origin    Show Yahtzotron's origin story.
  play      Play a game against Yahtzotron.
  train     Train a new model through self-play.
```

Why don't you try a game against one of the pre-trained agents?

```bash
$ yahtzotron play pretrained/yahtzee-score.pkl
```

#### Bonus

When you play Yahtzotron, it is going to tell you what its current strategy is before every action (to teach us puny humans how to play):

```
> My turn!
> Roll #1: [3, 3, 3, 5, 6].
> I think I should go for Threes, so I'm keeping [3, 3, 3].
> Roll #2: [3, 3, 3, 3, 4].
> I think I should go for Threes or Yatzy, so I'm keeping [3, 3, 3, 3].
> Roll #3: [1, 3, 3, 3, 3].
> I'll pick the "Threes" category for that.
```
