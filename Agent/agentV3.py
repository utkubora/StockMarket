import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

import gym

# Initialize the environment
env = gym.make('CartPole-v1')


class LSTMAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.linear(lstm_out[:, -1, :])  # Take the output of the last sequence step
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

def train(model, env, episodes, gamma=0.99):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for episode in range(episodes):
        state = env.reset()
        state = np.expand_dims(state, 0)  # Add batch dimension
        rewards = []
        log_probs = []
        hidden = model.init_hidden()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state)
            probs, hidden = model(state_tensor, hidden)
            m = torch.distributions.Categorical(torch.softmax(probs, dim=-1))
            action = m.sample()
            log_prob = m.log_prob(action)
            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            state = np.expand_dims(next_state, 0)  # Update state and maintain batch dimension

        discounted_rewards = [gamma**i * r for i, r in enumerate(rewards)]
        discounted_rewards = torch.tensor(discounted_rewards)
        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Episode {episode+1}: Total Reward: {sum(rewards)}')

# Initialize model and environment
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
hidden_dim = 128

model = LSTMAgent(input_dim, hidden_dim, output_dim)

# Train the model
train(model, env, episodes=100)

