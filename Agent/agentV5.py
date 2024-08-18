import pandas as pd

data = pd.read_csv('etfs/ABEQ.csv', encoding='utf-8')

data['Return'] = data['Close'].pct_change().fillna(0)


import numpy as np

class TradingEnvironment:
    def __init__(self, data, initial_balance=1000, max_trades=100):
        self.data = data
        self.initial_balance = initial_balance
        self.max_trades = max_trades
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.total_trades = 0
        self.done = False
        self.history = []
        return self._get_state()

    def _get_state(self):
        # Ensure that the window includes exactly 10 days of returns
        window_size = 10  # Fixed window size
        
        window = self.data[self.current_step:self.current_step + window_size]['Return'].values
        if len(window) < window_size:
            # If less than window_size, pad with zeros (or you could handle differently)
            window = np.append(window, np.zeros(window_size - len(window)))
        return np.append(window, [self.balance])


    def step(self, action):
        # 0=Hold, 1=Buy, 2=Sell
        print(self.current_step)
        print(self.data.shape)
        try:
            current_price = self.data.iloc[self.current_step]['Close']
            self.current_step += 1
        except:
            current_price = self.data.iloc[-1]['Close']
            self.done = True

        if action == 1 and self.balance >= current_price:  # Buy
            self.stock_owned += 1
            self.balance -= current_price
            self.total_trades += 1
        elif action == 2 and self.stock_owned > 0:  # Sell
            self.stock_owned -= 1
            self.balance += current_price
            self.total_trades += 1

        # Calculate new value of portfolio
        portfolio_value = self.balance + self.stock_owned * current_price
        reward = portfolio_value - self.initial_balance

        if self.current_step >= len(self.data) or self.total_trades > self.max_trades:
            self.done = True

        next_state = self._get_state()
        
        return next_state, reward, self.done

# Initialize the environment with the preprocessed data
env = TradingEnvironment(data)


import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize DQN
state_size = 11  # number of recent returns considered
action_size = 3  # actions: hold, buy, sell
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

from collections import deque
import numpy as np
import random

# Parameters
batch_size = 32
n_episodes = 500  # number of episodes for training
gamma = 0.95  # discount factor

# Epsilon-greedy parameters
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

# Replay buffer
replay_buffer = deque(maxlen=2000)

# Training loop
for episode in range(n_episodes):
    state = torch.randn(state_size)  # an example initial state
    state = state.unsqueeze(0)
    for t in range(200):  # number of time steps
        if random.random() <= epsilon:
            action = random.randrange(action_size)
        else:
            action_values = model(state)
            action = torch.argmax(action_values).item()

        # Simulate environment step (to be implemented)
        next_state, reward, done = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        
        # Before appending to replay buffer
        state_shape = state.shape
        next_state_shape = next_state.shape
        print("------------------------------------------------------------")
        
        print(state)
        print(next_state)
        
        print("------------------------------------------------------------")
        
        print(state_shape)
        print(next_state_shape)
        
        print("------------------------------------------------------------")
        
        if state_shape[1] != 11 or next_state_shape[1] != 11:
            print(f"Error in state dimension: state_shape {state_shape}, next_state_shape {next_state_shape}")
            continue  # Skip this transition or handle error
        # Store the transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            break

        # Experience replay
        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            # Check shapes before concatenation
            if any(s.shape[1] != 11 for s in states) or any(s.shape[1] != 11 for s in next_states):
                print("Inconsistent state size found!")
                continue  # handle the issue or debug further

            states = torch.cat(states)
            next_states = torch.cat(next_states)

            # Compute Q values
            
            print(f"tt unsqueze {(torch.Tensor(actions)).unsqueeze(1)}")
            
            current_q_values = (model(states).gather(1, (torch.Tensor(actions)).unsqueeze(1) )).squeeze(1)
            next_q_values = model(next_states).max(1)[0]
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)

            # Loss and backpropagation
            loss = loss_fn(current_q_values, expected_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training completed.")