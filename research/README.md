# Playing strength: how to measure and how to play

Move-prediction accuracy (top-1) only says how human-like the policy is; it does
not say whether the model can win. Everything here measures **real play strength**
by playing games, and provides the project's usable player.

## The player: `play.py`

The raw supervised policy (argmax) is far too weak to play directly — it is 1-ply
blind and hangs pieces. The usable player wraps the policy in a full-width
material negamax + capture-quiescence search (`arena.PolicySearchAgent`); the
network only orders and tie-breaks moves, the search decides.

    python research/play.py                  # you play White
    python research/play.py --color black    # you play Black
    python research/play.py --depth 3        # stronger, slower

## The benchmark: `arena.py`

Plays the models against three opponents of increasing strength and reports
win/draw/loss scores:

- `random`            — sanity bar (a legal mover). >=90% = PASS.
- `greedy-material`   — 1-ply material greedy. >50% = usable club-level prior.
- `minimax-d2`        — full-width depth-2 material minimax (`--hard-opp`), a
                        proxy for a real beginner engine.

    python research/arena.py --games 20 --search --hard-opp --no-mcts

`--search` adds the policy-guided search agent; `--search-depth` / `--qdepth`
control the negamax plies and the quiescence plies.

## Measured results (2026-10-15, 17-plane encoding, MIN_ELO=1800 data)

| agent | vs random | vs greedy | vs minimax-d2 |
|---|---|---|---|
| supervised policy (argmax) | ~72% | ~15% | ~0% |
| **supervised policy + search (d2,q4)** | **100%** | **~98%** | **~97%** |
| RL policy (stage 10) | ~60% | ~15% | ~2% |

Read this honestly:

- The **policy + search player is the deliverable** and clearly beats a depth-2
  material engine, a 1-ply greedy mover and a random mover.
- The **network alone is a weak player** (argmax loses to the depth-2 engine).
  Supervised imitation of human moves blunders under argmax; raising the training
  ELO floor did not fix this, so the supervised path has plateaued as a
  weak-but-useful prior.
- Most of the player's strength is the classical search; the network contributes
  move ordering / tie-breaks. Making the *network* carry the strength requires a
  working value head + MCTS over multiple self-play rounds (the AlphaZero loop).

## Distillation attempt: failed (do not use)

`distill.py` tried one round of policy iteration — label positions with the
search's chosen move and fine-tune the policy head on them (AlphaZero's
policy-target idea, with classical search standing in for MCTS). With a weak
teacher (depth 1, qdepth 3) and 2 epochs at lr 1e-4 it **regressed everything**:

| agent | before distill | after distill |
|---|---|---|
| policy vs random / greedy / minimax-d2 | 72.5% / 15% / 0% | 47.5% / 5% / 2.5% |
| search vs random / greedy / minimax-d2 | 100% / 97.5% / 97.5% | 97.5% / 97.5% / 80% |

The too-shallow teacher's labels corrupted a prior that was already better than
them, and the damage propagated into the search player. Original weights are
preserved at `best_model_pre_distill.pth`; restore them before using `play.py`.
Distillation is only worth retrying with a teacher at least as strong as the
target (full-width d2+q4 search, or real MCTS) and a lower learning rate.

## Known gap / future work

Stage 10 RL is currently **net-negative** (the RL policy scores below the
supervised policy): one small self-play round leaves the value head near noise,
and MCTS then overrides a decent policy with garbage values. To make the neural
component carry real strength: run several policy-improvement (self-play)
iterations with enough games that value targets are non-degenerate, and only
adopt an RL checkpoint when arena shows `rl-mcts` beating `supervised-search`.
