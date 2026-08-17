import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from helper import plot
import pickle
import os



class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0     
        self.gamma = 0.9     
        self.lr = 0.1        
        self.q_table = {}   

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),


            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),


            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

  
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x, 
            game.food.x > game.head.x, 
            game.food.y < game.head.y,  
            game.food.y > game.head.y  
        ]

        return tuple(int(x) for x in state)  

    def get_action(self, state):
        self.epsilon = max(10, 80 - self.n_game) 
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) 
        else:
            if state not in self.q_table:
                self.q_table[state] = [0, 0, 0]
            move = int(np.argmax(self.q_table[state]))  

        final_move[move] = 1
        return final_move

    def update_q_table(self, state, action_idx, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]

        old_q = self.q_table[state][action_idx]
        max_next_q = max(self.q_table[next_state])

    
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q * (1 - int(done)) - old_q)
        self.q_table[state][action_idx] = new_q


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            action_idx = final_move.index(1)
            agent.update_q_table(state_old, action_idx, reward, state_new, done)

            if done:
                game.reset()
                agent.n_game += 1
                total_score += score

                if score > record:
                    record = score

                print('Game', agent.n_game, 'Score', score, 'Record', record)

                plot_scores.append(score)
                mean_score = total_score / agent.n_game
                plot_mean_score.append(mean_score)
                plot(plot_scores, plot_mean_score)
    

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving Q-table and plot...")

        plot(plot_scores, plot_mean_score, mode="tabular_Q", n_games=agent.n_game)
        print(f"Plot saved to plots/tabular_Q_{agent.n_game}games.png")

        folder = "tabular_models"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filename = f"q_table_{agent.n_game}games.pkl"
        file_path = os.path.join(folder, filename)
        
        with open(file_path, "wb") as f:
            pickle.dump(agent.q_table, f)

        print(f" Q-table saved at: {file_path}")

        
        

if __name__ == '__main__':
    train()
