import pytest

import snake_game
from snake_game import Direction, Point, BLOCK_SIZE
from tests.conftest import place_snake

STRAIGHT, RIGHT_TURN, LEFT_TURN = [1, 0, 0], [0, 1, 0], [0, 0, 1]


def test_reset_state(game):
    assert game.direction == Direction.RIGHT
    assert game.head == Point(320, 240)
    assert game.snake == [Point(320, 240), Point(300, 240), Point(280, 240)]
    assert game.score == 0
    assert game.frame_iteration == 0
    assert game.food not in game.snake


@pytest.mark.parametrize('action, direction, head', [
    (STRAIGHT, Direction.RIGHT, Point(340, 240)),
    (RIGHT_TURN, Direction.DOWN, Point(320, 260)),
    (LEFT_TURN, Direction.UP, Point(320, 220)),
])
def test_move_is_relative_to_current_direction(game, action, direction, head):
    game._move(action)
    assert game.direction == direction
    assert game.head == head


def test_four_right_turns_return_to_start_direction(game):
    for _ in range(4):
        game._move(RIGHT_TURN)
    assert game.direction == Direction.RIGHT


@pytest.mark.parametrize('pt, expected', [
    (Point(620, 240), False),  # last column
    (Point(640, 240), True),   # past right wall
    (Point(-20, 240), True),   # past left wall
    (Point(0, 460), False),    # last row
    (Point(0, 480), True),     # past bottom wall
    (Point(0, -20), True),     # past top wall
])
def test_is_collision_with_walls(game, pt, expected):
    assert game.is_collision(pt) is expected


def test_is_collision_with_body(game):
    assert game.is_collision(game.snake[1]) is True


def test_is_collision_defaults_to_head(game):
    assert game.is_collision() is False
    place_snake(game, Point(640, 240), Direction.RIGHT)
    assert game.is_collision() is True


def test_is_collision_checks_given_point_not_head(game):
    # Regression: is_collision(pt) used to ignore pt and always test the head.
    assert game.is_collision() is False
    assert game.is_collision(Point(-BLOCK_SIZE, 0)) is True


def test_play_step_moves_without_growing(game):
    game.food = Point(0, 0)
    reward, done, score = game.play_step(STRAIGHT)
    assert (reward, done, score) == (0, False, 0)
    assert game.snake == [Point(340, 240), Point(320, 240), Point(300, 240)]


def test_play_step_eating_food_grows_and_rewards(game):
    game.food = Point(340, 240)
    reward, done, score = game.play_step(STRAIGHT)
    assert (reward, done, score) == (10, False, 1)
    assert len(game.snake) == 4
    assert game.food not in game.snake


def test_play_step_hitting_wall_ends_game(game):
    place_snake(game, Point(620, 240), Direction.RIGHT)
    game.food = Point(0, 0)
    reward, done, score = game.play_step(STRAIGHT)
    assert (reward, done, score) == (-10, True, 0)


def test_play_step_times_out_when_snake_dawdles(game):
    game.food = Point(0, 0)
    game.frame_iteration = 100 * (len(game.snake) + 1)
    reward, done, _ = game.play_step(STRAIGHT)
    assert (reward, done) == (-10, True)


def test_place_food_retries_when_landing_on_snake(game, monkeypatch):
    # First draw lands on the head (320, 240); the second is free.
    draws = iter([16, 12, 0, 0])
    monkeypatch.setattr(snake_game.random, 'randint', lambda a, b: next(draws))
    game._place_food()
    assert game.food == Point(0, 0)


def test_reset_after_game_over(game):
    place_snake(game, Point(620, 240), Direction.RIGHT)
    game.score = 5
    game.reset()
    assert game.score == 0
    assert game.head == Point(320, 240)
