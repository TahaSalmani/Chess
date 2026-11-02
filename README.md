# Chess — a policy network + classical search player, trained on human games

This project trains a chess **policy** network on human games and wraps it in a
full-width material search to produce a playable opponent. It ships as a local
terminal player, a zero-dependency browser player (graphical board, pieces and
synthesised sound), and a Docker image deployed to Render.

**Live demo:** <https://chess-web-onrender-com.onrender.com/> (you play White;
the first request after idle can take ~50 s while the free instance wakes up).

## The honest headline

Move-prediction accuracy only says how *human-like* the policy is. It does not
say whether the model can win. So this repo measures strength by playing games
(`research/arena.py`) and reports the numbers straight:

| agent | vs random | vs greedy-material | vs minimax-d2 |
|---|---|---|---|
| supervised policy (argmax) | ~72% | ~15% | ~0% |
| **policy + search (depth 2, q4)** | **100%** | **~98%** | **~97%** |
| RL policy (stage 10) | ~60% | ~15% | ~2% |

The **policy + search player is the deliverable**. The network alone is a weak
player — imitation of human moves blunders badly under argmax. Most of the
playing strength is the classical negamax + capture-quiescence search; the
network contributes move ordering and tie-breaks. Making the *network* carry
the strength needs a working value head and several rounds of MCTS self-play,
which two rejected rounds showed is out of reach on a free-GPU budget
([the RL chapter is closed](#the-rl-chapter-closed)).

Full results, methodology and the failed distillation experiment:
[`research/README.md`](research/README.md).

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-deploy.txt
```

(torch is installed first and separately because the CPU wheel index does not
carry the other packages; this is the same order the `Dockerfile` uses.)

Play in the browser (opens a local page with a graphical board):

```powershell
python research\web_player.py            # http://127.0.0.1:8765
python research\web_player.py --color black --depth 3
```

Click a piece, then a highlighted square. The agent answers automatically.
While it thinks the board is locked, so double-clicks cannot produce spurious
"illegal move" errors; if the page and server ever disagree, the page silently
re-syncs from `GET /api/state` instead of showing a dialog.

Play in the terminal:

```powershell
python research\play.py                  # you play White
python research\play.py --depth 3 --qdepth 4
```

Moves are accepted as SAN (`e4`, `Nf3`, `O-O`) or UCI (`e2e4`).

Flags shared by both players: `--color`, `--depth` (full-width negamax plies),
`--qdepth` (capture-quiescence plies at the leaves), `--weights` (defaults to
`artifacts/prepare_torch_callbacks/best_model.pth`), and for the web player
`--port` / `--no-open`.

## How a move is chosen

`research/arena.PolicySearchAgent` is the engine both players use:

1. The network scores all 4096 `from_square*64 + to_square` actions; legal moves
   are ordered by policy score for alpha-beta efficiency. **The policy never
   prunes a move** — a competent search can override a weak prior.
2. A full-width negamax runs to `--depth` over every legal move, dropping into
   capture-only quiescence to `--qdepth` so the leaves are not tactically blind.
   Leaves are scored by material (`P1 N3 B3 R5 Q9`), with MVV-LVA ordering and a
   narrowing root window (bound ties are re-searched exactly, which provably
   yields the same tie set as a full-window search — it is a pure speed change).
3. Mate is scored by distance, stalemate as `-0.5` (never stalemate a win), and
   stepping into a repetition as `0.0` (a material-only leaf cannot tell a
   shuffle from progress, and the agent used to repeat checks forever).
4. Moves the search scores identically are sampled with the policy prior, so
   games are not identical replays but the play stays human-like.

## Repository layout

```
src/Chess/
  components/     encoding, dataset, torch policy net, value+policy net, MCTS,
                  self-play, RL trainer, data ingestion/validation/transformation
  config/         ConfigurationManager — reads config/config.yaml + params.yaml
  entity/         dataclasses for the config boxes
  pipeline/       stage_01 ... stage_10 entry points (one per DVC stage)
  utils/          file/dir helpers used by every stage
research/         arena.py, play.py, web_player.py, rl_gate.py, benchmarks, tests
config/config.yaml   paths (artifacts dirs, file names)
params.yaml          numbers (channels, layers, lr, epochs, MCTS settings)
dvc.yaml             the reproducible pipeline
artifacts/           pipeline outputs; best_model.pth is committed
templates/index.html early Flask-era page (not used by web_player.py)
Dockerfile           image that runs research/web_player.py
.github/workflows/   docker-publish.yml (image build), ci.yaml (import smoke test)
```

Two config files on purpose: `config/config.yaml` holds *paths*, `params.yaml`
holds *numbers*. `src/Chess/config/configuration.py` merges them into boxes and
logs the resolved values, so a run is reproducible from the two YAML files.

Note that stages 04/05/06 (`PrepareBaseModel`, `TrainModel`, `ModelEvaluation`)
are the original TensorFlow/Keras path. They are **not in `dvc.yaml`** and are
kept only for history; the live pipeline is 01 → 02 → 03 → 07 → 08 → 09 → 10.

## The data pipeline

| stage | file | what it does |
|---|---|---|
| 01 DataIngestion | `stage_01_DataIngestion.py` | downloads Lichess DB games (`fetch_pgn_sample.py`), keeps `MIN_ELO=1800` |
| 02 DataValidation | `stage_02_DataValidation.py` | schema/column checks |
| 03 DataTransformation | `stage_03_DataTransformation.py` | PGN → (position, move) pairs |
| 07 TorchDataset | `stage_07_TorchDataset.py` | encodes positions and writes train/val splits |
| 08 TorchBaseModel | `stage_08_TorchBaseModel.py` | builds the policy network |
| 09 Train Torch Model | `stage_09_Train_Torch_Model.py` | trains it with callbacks; writes `best_model.pth` |
| 10 Pytorch RL | `stage_10_Pytorch_RL.py` | batched MCTS self-play → train loop (see below) |

Run it with `dvc repro` (optionally `dvc repro <stage_name>`). The DVC remote is
a Google Drive path (`.dvc/config`), so in practice training is run in a
notebook — `kaggle_full_run.ipynb`, `kaggle_dl`, `colab_full_run.ipynb`,
`colab_pipeline.ipynb`, `colab_stage10.ipynb` are the paste-ready versions.
Because `requirements.txt` does not pin torch, install it separately as above.

### Position encoding

17 planes of 8×8 (`IMAGE_SIZE: [8, 8, 17]`): piece-presence planes for both
colours plus castling/turn context. The label is a single index
`from_square * 64 + to_square` over `CLASSES: 4096`, so **underpromotions
collapse onto the queen move** — a known simplification, not a bug.

### The shipped checkpoint

`artifacts/prepare_torch_callbacks/best_model.pth` is the only committed weight
file (`.gitignore` allows just that path, so deployment needs no artifact
download). It has **4 transformer layers** while `params.yaml` says
`NUM_LAYERS: 10`; `research/play.load_agent` detects the mismatch and adapts the
network to the checkpoint, printing
`adapting net to checkpoint: IN_CHANNELS=17, NUM_LAYERS=4`. Architecture:
`D_MODEL=128`, `NUM_HEADS=4`, `FF_DIM=512`, `IN_CHANNELS=17`. Training ran with
`LR=0.0015`, `BATCH_SIZE=256`, `EPOCHS=100`.

## The RL chapter (closed)

Stage 10 runs a real AlphaZero-style loop: batched MCTS self-play → train,
repeated `RL_ITERATIONS` times, with Dirichlet root noise, visit-count
temperature sampling and a resignation threshold. Adoption is decided by
`research/rl_gate.py`, which plays the RL candidate (MCTS) against the
supervised policy+search champion with alternating colours and only promotes it
to a separate `promoted_model.pth`. It never touches `best_model.pth`
(the SHA is asserted before and after).

Two full rounds were run on Kaggle GPU and **both were rejected**:

| round | change | gate (rl-mcts vs supervised-search) | decision |
|---|---|---|---|
| 1 | bootstrap from supervised, 3 iters × 32 games | W0/D5/L5 = 25% | REJECT |
| 2 | fresh value head + resignation threshold | W0/D0/L20 = 0% | REJECT |

Round 1's self-play pool was ~94% move-cap draws, so value targets were almost
all `0.0`. Round 2 added resignation to convert dead draws into ±1 signal, and
the gate got *worse*. The read: a near-noise value head actively misleads PUCT
below the classical search it is wrapped around. This is chicken-and-egg — the
value head needs a strong policy to produce meaningful games, and the supervised
policy is too weak to provide them. More rounds or threshold tuning will not fix
it at this scale. `research/distill.py` (one round of policy iteration with a
too-shallow teacher) also regressed everything and is documented as a failure in
`research/README.md`; original weights are preserved and must be restored before
using `play.py`.

## Measuring strength

```powershell
python research\arena.py --games 20 --search --hard-opp --no-mcts
```

Opponents, weakest to strongest: `random` (sanity bar, ≥90% = PASS),
`greedy-material` (1-ply material greedy, >50% = usable club-level prior),
`minimax-d2` (`--hard-opp`, full-width depth-2 material minimax — a proxy for a
real beginner engine). `--search` adds the policy-guided search agent;
`--search-depth` / `--qdepth` control its plies.

## Tests

```powershell
python research\test_rl_gate.py
python research\test_self_play.py
```

`.github/workflows/ci.yaml` currently only runs an import smoke test
(`python -c "from Chess import logger; ..."`) — these two files are not wired
into CI yet.

## Deployment

`Dockerfile` (python:3.12-slim, CPU torch wheel, weights baked in,
`ENV PORT=8765 HOST=0.0.0.0`, `CMD ["python", "research/web_player.py"]`)
→ `.github/workflows/docker-publish.yml` builds and pushes
`tahasalmani/chess-web:latest` on pushes to `main` and `perf/batched-mcts`
→ a Render Web Service "from Existing Image" runs it (Free tier, Oregon,
512 MB / 0.1 CPU).

**Free tier does not auto-pull.** After a successful Actions run you must open
the Render service → *Manual Deploy* → *Deploy latest image*, or the site keeps
serving the previous image. You can confirm which build is live by requesting
`GET /api/state`: a JSON `409` means the current build, HTML means an older one.

The server is a stdlib `ThreadingHTTPServer` with no external dependencies and
no image or audio assets — the board is CSS + Unicode glyphs and the sounds are
synthesised with WebAudio. Each browser gets its **own game** via an `sid`
cookie (HttpOnly, SameSite=Lax, 2 h TTL, max 200 sessions, pruned by TTL then
LRU). A global lock guards only the session dict and never a search; each game
has its own lock, so concurrent visitors do not slow each other down beyond the
single-CPU limit.

## Known limitations

- **The network is a prior, not the player.** Argmax loses to a depth-2 material
  engine. Raising the training ELO floor to 1800 did not fix this; the
  supervised path has plateaued as a weak-but-useful prior.
- **Material-only leaves.** No king safety, pawn structure, mobility or
  endgame knowledge. Consequently the agent **cannot force mate with a bare king
  plus minor pieces** and will shuffle in some endgames (repetition is scored as
  a draw, so it does not lose, but it does not win either). An endgame
  king-tropism term is the obvious next improvement.
- **Underpromotions are unreachable** because of the 4096-index action space.
- **Latency on the free tier.** Worst-case positions take ~6.6 s locally, which
  maps to roughly 100 s on 0.1 CPU — long enough to hit proxy timeouts. Typical
  moves are 2.3–3.0 s live. A per-search time budget is the planned fix.
- **No `LICENSE` file.** The repo is public but has no explicit licence terms.
- `template.py`, `templates/index.html` and pipeline stages 04/05/06 are dead
  code kept for history.

## Requirements

`requirements-deploy.txt` (playing only): python-chess, numpy, pyyaml,
python-box, ensure — plus torch, installed separately.
`requirements.txt` (full pipeline): adds pandas, notebook and the local package
(`pip install -e .`).
