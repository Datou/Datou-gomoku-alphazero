"""RIF standard opening patterns (first three stones) for self-play.

Renju's 26 standard openings are defined by Black's first stone (centre), White's
second stone (orthogonal = direct, diagonal = indirect) and Black's third stone
within two points of the centre.  Under the symmetry that keeps the first two
stones fixed these fall into exactly 13 direct and 13 indirect patterns.

The book contains the three-stone prefixes only.  Games start from one of them
and the model plays White's fourth move onward, so the model is trained from
balanced standard positions instead of the empty board.

Opening stones are recorded as normal training plies with a one-hot policy
target, which teaches the book itself; no rule or tactical override is added.
"""
from __future__ import annotations

import numpy as np

BOARD_SIZE = 15
CENTRE = (7, 7)
DIRECT_SECOND = (7, 8)
INDIRECT_SECOND = (8, 8)
MAX_THIRD_DISTANCE = 2


def _reflect_row(point):
    r, c = point
    return (14 - r, c)


def _transpose(point):
    r, c = point
    return (c, r)


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _third_candidates(second):
    candidates = []
    for r in range(CENTRE[0] - MAX_THIRD_DISTANCE, CENTRE[0] + MAX_THIRD_DISTANCE + 1):
        for c in range(CENTRE[1] - MAX_THIRD_DISTANCE, CENTRE[1] + MAX_THIRD_DISTANCE + 1):
            point = (r, c)
            if point in (CENTRE, second):
                continue
            if _chebyshev(point, CENTRE) <= MAX_THIRD_DISTANCE:
                candidates.append(point)
    return candidates


def _orbits(second, reflection):
    seen = set()
    openings = []
    for point in _third_candidates(second):
        if point in seen:
            continue
        orbit = {point, reflection(point)}
        seen |= orbit
        openings.append(tuple(sorted(orbit))[0])
    return sorted(openings)


def build_openings():
    """Return the 26 canonical three-stone opening prefixes as coordinate tuples."""
    direct = [(CENTRE, DIRECT_SECOND, third) for third in _orbits(DIRECT_SECOND, _reflect_row)]
    indirect = [(CENTRE, INDIRECT_SECOND, third) for third in _orbits(INDIRECT_SECOND, _transpose)]
    openings = direct + indirect
    assert len(openings) == 26, len(openings)
    return openings


RIF_OPENINGS = build_openings()

# The eight symmetries of the square that fix the centre.
_SYMMETRIES = (
    lambda r, c: (r, c),
    lambda r, c: (c, 14 - r),
    lambda r, c: (14 - r, 14 - c),
    lambda r, c: (14 - c, r),
    lambda r, c: (r, 14 - c),
    lambda r, c: (14 - r, c),
    lambda r, c: (c, r),
    lambda r, c: (14 - c, 14 - r),
)


def apply_symmetry(opening, index):
    transform = _SYMMETRIES[index % len(_SYMMETRIES)]
    return [transform(r, c) for r, c in opening]


def sample_opening(rng=None, rotate=True):
    """Sample one of the 26 openings, optionally under a random symmetry."""
    rng = np.random.default_rng() if rng is None else rng
    opening = RIF_OPENINGS[int(rng.integers(len(RIF_OPENINGS)))]
    if rotate:
        opening = apply_symmetry(opening, int(rng.integers(len(_SYMMETRIES))))
    return [tuple(point) for point in opening]


def choose_opening(rng, free_ratio=0.0):
    """Return the opening prefix for one self-play game.

    With probability ``free_ratio`` only Black's mandatory centre stone is
    placed and the model plays White's second move onward, so free openings stay
    in the training distribution. Otherwise one of the 26 standard RIF
    three-stone prefixes is sampled.
    """
    if not 0.0 <= free_ratio <= 1.0:
        raise ValueError('free_ratio must be within [0, 1].')
    if free_ratio > 0.0 and rng.random() < free_ratio:
        return [CENTRE]
    return sample_opening(rng)


def opening_actions(opening):
    return [r * BOARD_SIZE + c for r, c in opening]


def validate_opening(opening):
    """Structural checks; rule legality is checked by GomokuGame in tests."""
    if len(opening) != 3:
        raise ValueError('An opening prefix must contain exactly three stones.')
    if opening[0] != CENTRE:
        raise ValueError('Black must start at the centre.')
    if opening[2] in opening[:2]:
        raise ValueError('Opening stones must be distinct.')
    if _chebyshev(opening[2], CENTRE) > MAX_THIRD_DISTANCE:
        raise ValueError('Black third stone must be within two points of the centre.')
    return True
