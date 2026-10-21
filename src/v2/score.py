"""Score parsing (winner's perspective), e.g. '6-7(5) 7-6(2) [10-7]' or '6-2 3-1 RET'.

Returns None for anything that is not a recognisable score. A match tiebreak '[10-7]' counts as a set and as one game.
An unfinished set (like '3-1') is allowed only as the last set before RET/ABD/DEF; it adds games but no set.
"""
import re

SET = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\((\d+)\))?$")
MTB = re.compile(r"^\[(\d{1,2})-(\d{1,2})\]$")
STOP = {"RET", "ABD", "DEF", "Def."}


def _finished(a, b):
    hi, lo = max(a, b), min(a, b)
    return (hi == 6 and lo <= 4) or (hi == 7 and lo in (5, 6)) or (hi > 7 and hi - lo == 2)


def parse(score):
    """dict(games_w, games_l, sets_w, sets_l, tiebreaks, complete) or None if unparseable."""
    if not isinstance(score, str) or not score.strip():
        return None
    tokens = score.split()
    stop = tokens[-1] in STOP
    if stop:
        tokens = tokens[:-1]
    if not tokens or any(t in STOP for t in tokens):
        return None
    gw = gl = sw = sl = tb = 0
    for i, t in enumerate(tokens):
        m, k = SET.match(t), MTB.match(t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if not _finished(a, b):
                if not (stop and i == len(tokens) - 1) or max(a, b) > 7:
                    return None
            else:
                sw += a > b; sl += b > a; tb += {a, b} == {6, 7}
            gw += a; gl += b
        elif k:
            a, b = int(k.group(1)), int(k.group(2))
            if a == b:
                return None
            gw += a > b; gl += b > a; sw += a > b; sl += b > a; tb += 1
        else:
            return None
    if not stop and sw <= sl:
        return None  # in a completed match the winner must have won more sets
    return dict(games_w=gw, games_l=gl, sets_w=sw, sets_l=sl, tiebreaks=tb, complete=not stop)
