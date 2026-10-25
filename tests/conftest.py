import os
import sys

# Headless pygame and matplotlib: must be set before those modules are imported.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

# The project is a flat set of scripts, so make the repo root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import snake_game  # noqa: E402
from snake_game import Direction, Point, SnakeGameAI, BLOCK_SIZE  # noqa: E402


@pytest.fixture
def game(monkeypatch):
    # Don't throttle to the display frame rate during tests.
    monkeypatch.setattr(snake_game, 'SPEED', 0)
    return SnakeGameAI()


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    # model.save()/load() use ./model, so isolate them from the real checkpoint.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def place_snake(game, head, direction, body=None):
    """Put the snake at `head` facing `direction`, with a straight 3-block body behind it unless `body` is given."""
    game.direction = direction
    game.head = head
    if body is None:
        dx, dy = {
            Direction.RIGHT: (-BLOCK_SIZE, 0),
            Direction.LEFT: (BLOCK_SIZE, 0),
            Direction.UP: (0, BLOCK_SIZE),
            Direction.DOWN: (0, -BLOCK_SIZE),
        }[direction]
        body = [Point(head.x + dx, head.y + dy), Point(head.x + 2 * dx, head.y + 2 * dy)]
    game.snake = [head] + body
