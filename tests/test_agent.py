import numpy as np
import pytest
import torch

import agent as agent_module
from agent import Agent, MAX_MEMORY
from snake_game import Direction, Point, QuitRequested
from tests.conftest import place_snake

# State layout: [danger straight, danger right, danger left, dir L, R, U, D, food L, R, U, D]


@pytest.fixture
def agent(in_tmp_dir):
    torch.manual_seed(0)
    return Agent()


def test_fresh_agent(agent):
    assert agent.n_games == 0
    assert agent.record == 0


def test_discount_factor_is_nonzero(agent):
    # Regression: gamma was 0, so the agent ignored future rewards.
    assert agent.gamma == pytest.approx(0.9)
    assert agent.trainer.gamma == pytest.approx(0.9)


def test_agent_resumes_from_checkpoint(in_tmp_dir):
    Agent().model.save(record=7, n_games=42)
    resumed = Agent()
    assert (resumed.record, resumed.n_games) == (7, 42)


def test_state_at_start(agent, game):
    game.food = Point(0, 0)
    state = agent.get_state(game)
    assert state.dtype == int
    assert state.tolist() == [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]


def test_state_danger_straight_at_wall(agent, game):
    place_snake(game, Point(620, 240), Direction.RIGHT)
    assert agent.get_state(game)[:3].tolist() == [1, 0, 0]


def test_state_danger_left_and_straight_in_corner(agent, game):
    place_snake(game, Point(620, 0), Direction.RIGHT)
    assert agent.get_state(game)[:3].tolist() == [1, 0, 1]


def test_state_danger_right_when_heading_up_along_right_wall(agent, game):
    place_snake(game, Point(620, 240), Direction.UP)
    assert agent.get_state(game)[:3].tolist() == [0, 1, 0]


def test_state_detects_own_body(agent, game):
    # Heading up with part of the body directly to the right.
    body = [Point(320, 260), Point(340, 260), Point(340, 240), Point(340, 220)]
    place_snake(game, Point(320, 240), Direction.UP, body=body)
    assert agent.get_state(game)[:3].tolist() == [0, 1, 0]


def test_state_food_direction(agent, game):
    game.food = Point(600, 460)  # right of and below the head
    assert agent.get_state(game)[7:].tolist() == [0, 1, 0, 1]


def test_get_action_exploits_model_when_epsilon_exhausted(agent):
    agent.n_games = 200
    state = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0])
    move = agent.get_action(state)
    expected = torch.argmax(agent.model(torch.tensor(state, dtype=torch.float))).item()
    assert sum(move) == 1
    assert move.index(1) == expected


def test_get_action_explores_early_on(agent, monkeypatch):
    monkeypatch.setattr(agent_module.random, 'randint', lambda a, b: a)  # always roll lowest
    assert agent.get_action(np.zeros(11, dtype=int)) == [1, 0, 0]


def test_memory_is_bounded(agent):
    assert agent.memory.maxlen == MAX_MEMORY
    agent.remember(np.zeros(11), [1, 0, 0], 0, np.zeros(11), False)
    assert len(agent.memory) == 1


def test_train_long_memory_updates_weights(agent, game):
    for _ in range(5):
        state = agent.get_state(game)
        action = [1, 0, 0]
        reward, done, _ = game.play_step(action)
        agent.remember(state, action, reward, agent.get_state(game), done)

    before = [p.detach().clone() for p in agent.model.parameters()]
    agent.train_long_memory()
    assert any(not torch.equal(b, a) for b, a in zip(before, agent.model.parameters()))


@pytest.mark.parametrize('exc', [QuitRequested, KeyboardInterrupt])
def test_train_stops_cleanly_on_quit(in_tmp_dir, monkeypatch, exc):
    def stop(self, action):
        raise exc
    monkeypatch.setattr(agent_module.SnakeGameAI, 'play_step', stop)
    closed = []
    monkeypatch.setattr(agent_module.SnakeGameAI, 'close', lambda self: closed.append('game'))
    monkeypatch.setattr(agent_module, 'close_plot', lambda: closed.append('plot'))

    agent_module.train()  # returns instead of raising

    assert closed == ['game', 'plot']
