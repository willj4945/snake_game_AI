# Snake Game AI

[![CI](https://github.com/willj4945/snake_game_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/willj4945/snake_game_AI/actions/workflows/ci.yml)
[![Security](https://github.com/willj4945/snake_game_AI/actions/workflows/security.yml/badge.svg)](https://github.com/willj4945/snake_game_AI/actions/workflows/security.yml)

An agent that teaches itself to play Snake with **Deep Q-Learning**. The game is built with [Pygame](https://www.pygame.org/), the Q-network with [PyTorch](https://pytorch.org/), and training progress is plotted live with matplotlib.

## Quick start

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/willj4945/snake_game_AI.git
cd snake_game_AI
uv sync              # creates .venv with the locked dependencies (CPU-only PyTorch)
uv run agent.py      # start training
```

Two windows open: the game, and a chart of each game's score with the running mean. Training runs until you close the game window, press Esc or Q in it, or press Ctrl+C in the terminal. Any of these closes both windows and prints a summary. The checkpoint is only written when a new record is set, so stopping never overwrites it.

Training needs a display. On WSL, use WSLg (Windows 11) or an X server.

### Resuming and resetting

Whenever the agent beats its all-time record, it saves a checkpoint to `model/model.pth`. The checkpoint holds the network weights, the record, and the number of games played. On the next run the agent picks up where it left off. To train from scratch, delete `model/model.pth`.

The `model/` directory is gitignored, so checkpoints stay on your machine and a fresh clone starts with an untrained agent.

## How it works

Each frame, the agent looks at the game, picks a move, and learns from the result.

**State:** 11 yes/no values.

| Values | Meaning |
|---|---|
| 3 | Danger if the snake goes straight, turns right, or turns left (wall or its own body one block away) |
| 4 | Current direction: left, right, up, down |
| 4 | Food position relative to the head: left, right, above, below |

**Action:** go straight, turn right, or turn left (relative to the current heading).

**Reward:** +10 for eating food, −10 for dying. The game also ends if the snake goes `100 × length` frames without dying, so it can't circle forever.

**Model:** a small fully connected network (11 → 256 → 3) estimates the Q-value of each action. It is trained with the Bellman update `Q = r + γ · max Q(next state)`, with γ = 0.9, using Adam and MSE loss.

**Training:**
- *Short memory:* train on every transition as it happens.
- *Long memory:* after each game, train on a random batch of up to 1,000 past transitions from a replay buffer of 100,000.
- *Exploration:* epsilon-greedy. Random moves are common at first and stop after 80 games.

## Project layout

```
agent.py         Agent (state, action selection, replay memory) and the training loop (entry point)
model.py         Q-network (Linear_QNet) with checkpoint save/load, and the trainer (QTrainer)
snake_game.py    Snake environment (SnakeGameAI)
helper.py        Live training plot
model/           Saved checkpoint (gitignored, created on first record)
tests/           pytest suite
```

## Development

```bash
uv run pytest                                # run the tests (headless, no window opens)
uv run bandit -c pyproject.toml -r .         # security lint
```

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. To add or upgrade a package, use `uv add <pkg>` or `uv lock --upgrade-package <pkg>`. Commit the updated `uv.lock`.

PyTorch comes from the official CPU-only index (`[tool.uv.sources]` in `pyproject.toml`). The network is tiny, so a GPU doesn't help, and the CPU build is about 3 GB smaller than the CUDA build.

## CI and security

GitHub Actions runs:

| Workflow | Checks |
|---|---|
| **CI** (`ci.yml`) | Tests on Python 3.12 and 3.13, and verifies `uv.lock` is in sync with `pyproject.toml`. |
| **Security** (`security.yml`), on every push/PR and weekly | [pip-audit](https://github.com/pypa/pip-audit) checks locked dependencies for known CVEs. [Bandit](https://github.com/PyCQA/bandit) checks the code for insecure patterns. [CodeQL](https://codeql.github.com/) analyzes the Python code and the workflows themselves. |
| **Dependabot** (`dependabot.yml`) | Opens weekly PRs to update Python dependencies and GitHub Actions. |

Hardening in place:
- Workflows use read-only tokens by default.
- Third-party actions are pinned to full commit SHAs.
- Dependencies are locked.
- Only `torch` may come from the PyTorch index, which prevents dependency confusion.
- Checkpoints load with `torch.load(weights_only=True)`, so a tampered `.pth` file can't run code.
