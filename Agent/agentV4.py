import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Environment setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Hyperparameters
learning_rate = 0.01
gamma = 0.99  # Discount factor for past rewards
epsilon = 0.1  # Epsilon-greedy parameter
buffer_size = 10000
batch_size = 64

# Model, optimizer, and memory
model = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
memory = deque(maxlen=buffer_size)

def choose_action(state, epsilon):
    if state is None or len(state) == 0:
        raise ValueError("Received invalid state: {}".format(state))
    if random.random() > epsilon:
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        q_values = model(state)
        action = q_values.max(1)[1].item()
    else:
        action = env.action_space.sample()
    return action

def update_model(batch_size):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = model(next_states).max(1)[0]
    expected_q = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(current_q, expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        update_model(batch_size)
    print(f"Episode: {episode}, Total Reward: {total_reward}")
