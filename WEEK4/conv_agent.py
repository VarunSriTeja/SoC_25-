import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torchvision.transforms as T
from conv_model import ConvDQN
from game_image import get_screen_image

class ConvDQNAgent:
    def __init__(self, input_shape=(1, 40, 40), num_actions=3, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=50000, batch_size=32, min_epsilon=0.01):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvDQN(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((input_shape[1], input_shape[2])),
            T.ToTensor()
        ])

    def preprocess(self, state_img):
        # state_img: numpy array (H, W, 3), RGB
        img = self.transform(state_img)
        return img.unsqueeze(0).to(self.device)  # shape: (1, 1, H, W)

    def act(self, state_img):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        state = self.preprocess(state_img)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.cat([self.preprocess(s).float() for s,_,_,_,_ in minibatch]).to(self.device)
        actions = torch.tensor([a for _,a,_,_,_ in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([r for _,_,r,_,_ in minibatch], dtype=torch.float).to(self.device)
        next_states = torch.cat([self.preprocess(ns).float() for _,_,_,ns,_ in minibatch]).to(self.device)
        dones = torch.tensor([d for _,_,_,_,d in minibatch], dtype=torch.bool).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (~dones)
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)

    def get_state_from_game(self, game):
        return get_screen_image(game) 