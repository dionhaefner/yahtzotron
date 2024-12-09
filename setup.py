from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    # metadata
    name="yahtzotron",
    description="Beats you in Yahtzee.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dionhaefner/yahtzotron",
    author="Dion Häfner",
    author_email="mail@dionhaefner.de",
    # module
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.6",
    use_scm_version={"write_to": "yahtzotron/_version.py"},
    # dependencies
    setup_requires=["setuptools_scm"],
    install_requires=[
        "click",
        "numpy",
        "loguru",
        "jax",
        "flax",
        "dm-haiku",
        "rlax",
        "optax",
        "tqdm",
        "asciimatics",
    ],
    # CLI
    entry_points="""
        [console_scripts]
        yahtzotron=yahtzotron.cli:cli
    """,
)
