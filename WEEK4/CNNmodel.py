import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class ConvDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ConvDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.calc_conv_output(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def calc_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x).view(x.size()[0], -1)
        return self.fc_layers(conv_out)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
          state = torch.unsqueeze(state, 0)  
          next_state = torch.unsqueeze(next_state, 0)
          action = torch.unsqueeze(action, 0)
          reward = torch.unsqueeze(reward, 0)
          done = (done, )


        pred = self.model(state)

        target = pred.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][action[idx].item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



