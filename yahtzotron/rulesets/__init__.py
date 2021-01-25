from .yatzy import yatzy_rules

AVAILABLE_RULESETS = {r.name: r for r in (yatzy_rules,)}
