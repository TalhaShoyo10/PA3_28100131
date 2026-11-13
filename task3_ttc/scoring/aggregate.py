import numpy as np
from scipy.stats import rankdata

TIE_TOLERANCE = 1e-9


def earliest_best(values):
    best = np.max(values)
    return int(np.flatnonzero(values == best)[0])


def highest_score(scores):
    return earliest_best(np.asarray(scores, dtype=np.float64))


def weighted_vote(groups, weights):
    totals = {}
    first_seen = {}
    for position, (g, w) in enumerate(zip(groups, weights)):
        totals[g] = totals.get(g, 0.0) + float(w)
        first_seen.setdefault(g, position)
    best = max(totals.values())
    tied = [g for g, t in totals.items() if best - t <= TIE_TOLERANCE]
    winner = min(tied, key=lambda g: first_seen[g])
    return winner, first_seen[winner], totals


def self_consistency(groups):
    winner, position, totals = weighted_vote(groups, np.ones(len(groups)))
    return position, totals


def rwc_weights(scores):
    n = len(scores)
    if n == 1:
        return np.ones(1)
    ranks = rankdata(-np.asarray(scores, dtype=np.float64), method="average")
    return (n - ranks) / (n - 1)


def rwc(groups, scores):
    if len(groups) == 1:
        return 0, {groups[0]: 1.0}
    winner, position, totals = weighted_vote(groups, rwc_weights(scores))
    return position, totals


def vote_margin(totals):
    ordered = sorted(totals.values(), reverse=True)
    total = sum(ordered)
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    return (ordered[0] - runner_up) / total if total > 0 else 0.0
