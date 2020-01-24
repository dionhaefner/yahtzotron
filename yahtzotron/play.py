def print_sore(scorecard):
    colwidth = max(len(c.name) for c in scorecard.ruleset_.categories)
    separator_line = ''.join(['=' * colwidth, '+', '=' * 5])

    bonus_total, non_bonus_total = scorecard.score_summary()

    out = [
        '',
        separator_line
    ]

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if not cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = ''.join([' ', cat.name, ' | ', str(score) if filled else ''])
        out.append(line)

    out.append(separator_line)
    out.append(''.join([' Bonus ']))
    out.append(separator_line)

    for i, cat in enumerate(scorecard.ruleset_.categories):
        if cat.counts_towards_bonus:
            continue

        score = scorecard.scores[i]
        filled = scorecard.filled[i]

        line = ''.join([' ', cat.name, ' | ', str(score) if filled else ''])
        out.append(line)

    out.append(separator_line)
    out.append(''.join([' Total ', '|', str(bonus_total + non_bonus_total)]))
    out.append(separator_line)
    out.append('')

    return '\n'.join(out)
