"""Checks for adding a future, legally approved data source (not used yet: no extra source is merged).

A new source has its own player ids, so players must be linked by name before its matches can extend the ratings.
Anything ambiguous is reported, never guessed.
"""
import re
import unicodedata
from collections import defaultdict

from src.canonical import validate

SUFFIX = {"jr", "sr", "ii", "iii"}


def tokens(name):
    """Name -> lower-case ASCII words without punctuation or generational suffixes ('Felix Auger-Aliassime' -> felix auger aliassime)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("'", "").replace(".", "").replace("-", " "))
    return [t for t in s.split() if t not in SUFFIX]


def full_key(name):
    return " ".join(tokens(name))


def short_key(name):  # surname + first initial: fallback for spelling differences
    t = tokens(name)
    return f"{t[-1]}|{t[0][0]}" if t else ""


def map_players(external, known):
    """external, known: {id: name}. Returns (mapping ext_id -> known_id, unmatched ext_ids, ambiguous ext_ids)."""
    index = defaultdict(list)
    for kid, name in known.items():
        index[full_key(name)].append(kid)
    hits = {eid: index.get(full_key(name), []) for eid, name in external.items()}
    mapping = {eid: h[0] for eid, h in hits.items() if len(h) == 1}
    targets = defaultdict(list)
    for eid, kid in mapping.items():
        targets[kid].append(eid)
    clash = {eid for eids in targets.values() if len(eids) > 1 for eid in eids}  # two external players -> one known player
    ambiguous = sorted(clash | {eid for eid, h in hits.items() if len(h) > 1})
    return {e: k for e, k in mapping.items() if e not in clash}, sorted(e for e, h in hits.items() if not h), ambiguous


def validate_extension(new, base):
    """new, base: canonical tables. Raises if the new source is unsafe to append; returns a short report."""
    validate(new)
    assert not set(new.source) & set(base.source), "new rows must carry their own source name"
    assert not set(new.match_id) & set(base.match_id), "match ids collide with existing data"
    assert new.event_date.min() > base.event_date.max(), "new source overlaps existing dates: reconcile duplicates first"
    known = set(base.winner_id) | set(base.loser_id)
    players = set(new.winner_id) | set(new.loser_id)
    return {"matches": len(new), "players": len(players), "players_already_known": len(players & known),
            "first_date": new.event_date.min().date(), "last_date": new.event_date.max().date()}
