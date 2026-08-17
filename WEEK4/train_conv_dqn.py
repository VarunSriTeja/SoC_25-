import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode for pygame
import torch
from game import SnakeGameAI
from conv_agent import ConvDQNAgent
import numpy as np
from helper import plot

# Training parameters
NUM_EPISODES = 1000
SAVE_EVERY = 50
FRAME_SKIP = 4  # Agent acts every 4 frames
INPUT_SIZE = 40  # Use 40x40 for faster training

# NOTE: For maximum speed, set SPEED = 1000 or comment out self.clock.tick(SPEED) in game.py

if __name__ == '__main__':
    agent = ConvDQNAgent(input_shape=(1, INPUT_SIZE, INPUT_SIZE), num_actions=3)
    game = SnakeGameAI()
    scores = []
    mean_scores = []
    best_score = 0
    total_score = 0

    for episode in range(1, NUM_EPISODES + 1):
        game.reset()
        state = agent.get_state_from_game(game)
        done = False
        total_reward = 0
        step = 0
        action = agent.act(state)  # Initial action
        while not done:
            if step % FRAME_SKIP == 0:
                action = agent.act(state)
            reward, done, score = game.play_step([int(action == 0), int(action == 1), int(action == 2)])
            next_state = agent.get_state_from_game(game)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            step += 1
        scores.append(score)
        total_score += score
        mean_score = total_score / episode
        mean_scores.append(mean_score)
        plot(scores, mean_scores, mode="ConvDQN", n_games=episode)
        if score > best_score:
            best_score = score
            # Save best model
            model_folder = "conv_dqn_models"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            torch.save(agent.model.state_dict(), os.path.join(model_folder, f"best_model_ep{episode}.pth"))
        if episode % SAVE_EVERY == 0:
            print(f"Episode {episode}, Score: {score}, Mean: {mean_score:.2f}, Best: {best_score}, Epsilon: {agent.epsilon:.3f}")
    print("Training complete.") 