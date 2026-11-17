# renju_v2 — fresh Renju training with RIF balanced openings

Created 2026-11-16 Asia/Shanghai on explicit user request.

- Random initialization (no warm start from freestyle/clean_v2).
- Renju forbidden-move rules from step 0 (RULESET_ID renju_forbidden_v1).
- Every self-play game starts from one of the 26 RIF three-stone opening
  prefixes (rif_openings.py), randomly rotated/reflected (8 symmetries).
  The three opening stones are recorded as book plies with one-hot policy
  targets; the model plays White's fourth move onward.
- 20% of self-play games are free-opening games: only Black's mandatory centre
  stone is recorded, then MCTS plays White's second move onward
  (SELFPLAY_FREE_OPENING_RATIO=0.2). This keeps non-book openings in the
  training distribution.
- PLAY_TEMPERATURE_MOVES reduced 20 -> 8 after a 2026-11-15 audit showed
  White missed 30% of blockable fives because moves were sampled from a soft
  visit distribution (argmax would block 96%).
- renju_v1 is preserved for reference and is no longer trained.
