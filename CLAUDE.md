# CLAUDE.md

This file guides Claude Code (claude.ai/code) when working in this repository.

## Overview

A Snake game played by a Deep Q-Learning agent. Pygame renders the game, PyTorch provides the Q-network, and matplotlib plots training live. The project is a flat set of Python scripts, not an installable package. See `README.md` for the user-facing explanation.

## Commands

```bash
uv sync                                      # install locked deps into .venv (Python 3.12, CPU-only torch)
uv run agent.py                              # train (opens pygame + matplotlib windows; needs a display)
uv run pytest                                # all tests (headless)
uv run pytest tests/test_agent.py::test_state_at_start   # single test
uv run bandit -c pyproject.toml -r .         # security lint (same as CI)
```

Dependencies are managed with uv (`pyproject.toml` + `uv.lock`). There is no `requirements.txt`. Use `uv add` / `uv lock --upgrade-package` and commit `uv.lock`, because CI runs `uv sync --locked` and fails if the lock file is stale. `torch` is pinned to the PyTorch CPU index through `[tool.uv.sources]`, and that index is `explicit`, so no other package may resolve from it.

## Architecture

| File | Role |
|---|---|
| `snake_game.py` | `SnakeGameAI` environment. `play_step(action)` returns `(reward, game_over, score)`. Also defines `Direction`, `Point`, `BLOCK_SIZE=20`, and `SPEED=40` (FPS cap, which limits training speed). Calls `pygame.init()` and loads the font at import time. |
| `agent.py` | `Agent` builds the state, picks epsilon-greedy actions, and holds replay memory. `train()` is the main loop and the entry point. |
| `model.py` | `Linear_QNet` (11 → 256 → 3 MLP, ReLU; `save`/`load` checkpoints) and `QTrainer` (Adam + MSE Bellman update). `train_step` accepts a single transition or a batch. |
| `helper.py` | `plot(scores, mean_scores)` draws a live matplotlib chart through IPython display. |
| `model/model.pth` | Checkpoint dict `{'model_state', 'record', 'n_games'}`. It is written each time the score beats the all-time record. |

**Action encoding:** a one-hot list of length 3, `[straight, right turn, left turn]`, relative to the current direction (see `SnakeGameAI._move`).

**State (11 ints):** danger straight/right/left, current direction (L, R, U, D), food relative to head (left, right, up, down). It is built in `Agent.get_state` using `game.is_collision(pt)`. If you change the state size, update `Linear_QNet(11, ...)` in `Agent.__init__` and the tests.

**Rewards:** +10 for food, −10 for death or timeout. A timeout happens when `frame_iteration > 100 * len(snake)`.

**Training loop:** each step trains on the single transition ("short memory") and stores it in a deque capped at `MAX_MEMORY=100_000`. When an episode ends, the agent trains on a random batch of up to `BATCH_SIZE=1000` ("long memory"). The discount factor is `gamma=0.9`. Epsilon is `80 - n_games` out of 200, so after 80 games the agent makes no random moves.

**Checkpointing:** `Agent.__init__` calls `model.load()`, which restores weights, `record`, and `n_games`, so epsilon carries over between runs. A legacy bare-`state_dict` file still loads, with both counters reset to 0. `save()` and `load()` use the relative path `./model`.

## Tests

- `tests/conftest.py` sets `SDL_VIDEODRIVER=dummy` and `MPLBACKEND=Agg` before importing anything and puts the repo root on `sys.path`. New test files must import project modules *after* conftest runs (normal pytest behaviour).
- Fixtures:
  - `game`: a `SnakeGameAI` with `SPEED` patched to 0 so tests aren't throttled.
  - `in_tmp_dir`: chdir to `tmp_path` so `./model` checkpoints don't touch the real one.
  - Any test that constructs `Agent()` must use `in_tmp_dir`, because `Agent()` loads `./model/model.pth`.
- `place_snake(game, head, direction, body=None)` in conftest sets up board positions.
- The default board is 640×480, and the snake starts at head `(320, 240)` facing RIGHT.

## CI / security

- `.github/workflows/ci.yml` runs pytest on Python 3.12 and 3.13.
- `.github/workflows/security.yml` runs pip-audit, Bandit, and CodeQL on every push and PR, and weekly. pip-audit audits `uv export` output with `+cpu` stripped, because PyPI advisories only know the upstream torch version.
- Dependabot (`.github/dependabot.yml`) updates the `uv` and `github-actions` ecosystems.
- Rules to keep:
  - Pin actions to full commit SHAs with a `# vX.Y.Z` comment.
  - Keep `permissions: contents: read` as the default.
  - Keep `torch.load(..., weights_only=True)`.
  - Bandit skips only B311 (`random` is used for gameplay, not crypto).

## Pitfalls

- `.idea/` is committed (PyCharm project files, with the interpreter set to an old 3.9 venv). Ignore it.
- The game has no human-playable mode. `SnakeGameAI` expects actions from the agent.
- `model/model.pth` in git predates the `is_collision` fix, so its weights learned from broken danger inputs. Delete it to retrain cleanly.
