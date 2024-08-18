import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden=None):
        lstm_out, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output = self.fc(lstm_out.view(1, -1))
        return output, hidden

class RLAgent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, gamma=0.95):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gamma = gamma
        self.model = LSTMNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values, _ = self.model(state)
        action = torch.argmax(q_values).item()
        return action

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            q_values, _ = self.model(state)
            next_q_values, _ = self.model(next_state)
            q_value = q_values.squeeze(0)[action]
            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(next_q_values)
            loss = self.criterion(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
