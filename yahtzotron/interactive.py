def play_intro():
    from asciimatics.effects import Stars, Cog, Print, Effect
    from asciimatics.renderers import FigletText, StaticRenderer, SpeechBubble
    from asciimatics.scene import Scene
    from asciimatics.screen import Screen

    class Flash(Effect):
        def __init__(
            self, screen, rate, offset=0, duration=2, bg=Screen.COLOUR_WHITE, **kwargs
        ):
            super().__init__(screen, **kwargs)
            self._rate = rate
            self._duration = duration
            self._offset = offset
            self._bg = bg

        def reset(self):
            pass

        def _update(self, frame_no):
            if (frame_no - self._offset) % self._rate < self._duration:
                self._screen.clear_buffer(self._bg, 0, self._bg)

            elif (frame_no - self._offset) % self._rate == self._duration:
                self._screen.clear_buffer(7, 0, 0)

        @property
        def frame_update_count(self):
            return min(self._rate, self._duration)

        @property
        def stop_frame(self):
            return self._stop_frame

    man_standing = r"""
    ____
   (___ \
    oo~)/
   _\-_/_
  / \|/  \
 / / .- \ \
 \ \ .  /_/
  \/___(_/
   | |  |
   | |  |
   |_|__|
  (_(___]
    """

    robot = r"""
    ____
   [____]
   ]    [
 ___\__/___
|__|    |__|
 |_|_/\_|_|
 | | __ | |
 |_|[::]|_|
 \_|_||_|_/
   |_||_|
  _|_||_|_
 |___||___|
 """

    lab_left = r"""


   _______
  |ooooooo|
  |[]+++[]|
  |+ ___ +|
  |:|   |:|
  |:|___|:|
  |[]===[]|
_ ||||||||| _
  |_______|
    """

    lab_right = r"""
                |
                |
    ________    |
   | __  __ |   |
   |/  \/  \|   |
   |\__/\__/|   |
   |[][][][]|   |
   |++++++++|   |
   | ______ |   |
__ ||______|| __|
   |________|   \
                 \
                  \
                   \
                    \
    """

    def demo(screen):
        scenes = []

        # scene 1: splash screen
        effects = [
            Print(
                screen,
                FigletText("YAHTZOTRON", font="starwars"),
                int(screen.height / 2 - 8),
                transparent=False,
                speed=1,
            ),
            Print(
                screen,
                StaticRenderer(["The friendly robot that beats you in Yahtzee"]),
                int(screen.height / 2),
                speed=1,
            ),
            Cog(screen, 5, screen.height - 5, 8, colour=screen.COLOUR_YELLOW),
            Cog(
                screen,
                screen.width - 5,
                screen.height - 5,
                8,
                direction=-1,
                colour=screen.COLOUR_YELLOW,
            ),
            Stars(screen, 200),
            Print(
                screen,
                StaticRenderer(["🤖   🎲 🎲 🎲 🎲 🎲   🤖"]),
                int(screen.height / 2 + 4),
                speed=1,
            ),
        ]
        scenes.append(Scene(effects, 100))

        # scene 2: origin story
        effects = [
            Print(screen, StaticRenderer([lab_left]), x=5, y=0, speed=1),
            Print(screen, StaticRenderer([robot]), x=20, y=0, speed=1),
            Print(screen, StaticRenderer([man_standing]), x=40, y=0, speed=1),
            Print(screen, StaticRenderer([lab_right]), x=52, y=0, speed=1),
            Flash(screen, 25, duration=3, start_frame=25, stop_frame=35),
            Flash(screen, 30, duration=2, start_frame=30, stop_frame=35),
            Print(
                screen,
                StaticRenderer(["● ●"]),
                x=25,
                y=3,
                colour=screen.COLOUR_YELLOW,
                start_frame=30,
            ),
            Print(
                screen,
                SpeechBubble("It's alive!"),
                x=40,
                y=13,
                start_frame=50,
                stop_frame=75,
                clear=True,
                speed=1,
            ),
            Print(
                screen,
                SpeechBubble("State your prime directive"),
                x=35,
                y=13,
                start_frame=90,
                stop_frame=130,
                clear=True,
            ),
            Print(
                screen, SpeechBubble("T O . . . R O L L"), x=15, y=13, start_frame=150, clear=True, speed=1
            ),
        ]
        scenes.append(Scene(effects, 200))

        screen.play(scenes, repeat=False, stop_on_resize=True)

    Screen.wrapper(demo)


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
