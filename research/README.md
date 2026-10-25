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

## The browser player: `web_player.py`

The same agent, but as a local web app with a graphical board, pieces and sound
(no extra dependencies and no image/audio assets - the board is CSS + Unicode
glyphs and the sounds are synthesised with WebAudio):

    python research/web_player.py            # opens http://127.0.0.1:8765

Click a piece, then a highlighted square. The agent (supervised policy + search)
answers automatically; captures, check and game-end each play a distinct sound.
`--color black`, `--depth`, `--qdepth`, `--weights` and `--port` behave as in
`play.py`.

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

## The RL loop: two rounds, both rejected (closed)

Stage 10 runs a real AlphaZero-style loop (batched MCTS self-play -> train,
repeated `RL_ITERATIONS` times, with Dirichlet root noise, visit-count
temperature sampling and a resignation threshold). Adoption is decided by
`rl_gate.py`, which plays the RL candidate (MCTS) against the supervised
policy+search champion with alternating colours and only promotes it to a
separate `promoted_model.pth` on acceptance; it never touches `best_model.pth`
(sha-asserted before/after).

Two full rounds were run on Kaggle GPU and both were **rejected**:

| round | change | gate (rl-mcts vs supervised-search) | decision |
|---|---|---|---|
| 1 | bootstrap from supervised, 3 iters x 32 games | W0/D5/L5 = 25% | REJECT |
| 2 | fresh value head + resignation threshold | W0/D0/L20 = 0% | REJECT |

Round 1's self-play pool was ~94% move-cap draws, so value targets were almost
all 0.0. Round 2 added resignation (end a game once the search's own root value
stays below -threshold) to convert dead draws into +/-1 signal, but the gate got
*worse*, not better.

The honest read: the value head cannot bootstrap at this scale. MCTS's PUCT uses
the value to steer the search, so a near-noise value head actively misleads the
search below even the classical material search it is wrapped around (hence 0%).
This is a chicken-and-egg problem - the value head needs a strong policy to
produce meaningful games, and the supervised policy is too weak to provide them.
More self-play rounds or threshold tuning will not fix it; a real AlphaZero
value head needs orders of magnitude more self-play than a free-GPU budget
allows. **The RL path is closed; the deliverable remains policy + search.**

## Known gap / future work

The neural component is a weak-but-useful prior, not the source of strength:
most of the player's skill is the classical negamax + quiescence search. Making
the *network* carry the strength would require the value head + MCTS loop above
to actually work, which two rejected rounds show is out of reach at this scale.
If it is ever revisited, the entry point is several policy-improvement
iterations with enough games that value targets are non-degenerate, adopting a
checkpoint only when `rl_gate.py` shows `rl-mcts` beating `supervised-search`.
