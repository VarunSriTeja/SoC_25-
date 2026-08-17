import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from CNNmodel import Conv_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Conv_QNet(3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        grid_size = game.w // 20
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for pt in game.snake:
            grid[int(pt.y // 20)][int(pt.x // 20)] = 0.5

        grid[int(game.head.y // 20)][int(game.head.x // 20)] = 1.0
        grid[int(game.food.y // 20)][int(game.food.x // 20)] = -1.0

        return grid.reshape(1, grid_size, grid_size)


    def remember(self, state, action, reward, next_state, done):
        action_idx = np.argmax(action)
        self.memory.append((state, action_idx, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(
            np.array([state]), 
            np.array([action]), 
            np.array([reward]), 
            np.array([next_state]), 
            np.array([done])
        )


    def get_action(self, state):
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    print("Training started...")
    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_game += 1
                total_score += score
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save(f"CNN_{agent.n_game}games.pth")

                print('Game', agent.n_game, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                mean_score = total_score / agent.n_game
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        plot(plot_scores, plot_mean_scores, mode="CNN", n_games=agent.n_game)


if __name__ == '__main__':
    train()
