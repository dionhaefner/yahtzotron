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
