import torch
import numpy as np
import random
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning Rate


class Agent:
    def __int__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0  # control randomness
        self.gamma = 0    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Calls popleft()
        # TODO: Model, Trainer

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        pass


def train():
    pass


if __name__ == '__main__':
    train()
