import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Helper import *

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size: list, output_size):
        super().__init__()
        self.layers = []

        self.layers.append(nn.Linear(input_size, hidden_size[0]))

        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

        self.layers.append(nn.Linear(hidden_size[-1], output_size))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.layers[-1](x))
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        

class StockData:
    def __init__(self,dataset) -> None:
        self.day_index = 0
        self.data = dataset
        self.inventory = []
        self.score = 0
        
        self.actions = [[],[]]
        self.act_type = []
    
    def reset(self):
        self.day_index = 0
        self.inventory = []
        self.score = 0
        
        self.actions = [[],[]]
        self.act_type = []
        
    def play_step(self,fm):
        # burada kar zarar hesaplaması yapılmalı
        
        bought_price = 0,
        reward = 0

        if fm[0] == 1: # sell
            if len(self.inventory) == 0:
                reward = -10
                self.score += reward
                
            else:
                bought_price = self.inventory.pop(0)
                reward = max(self.data[self.day_index] - bought_price, 0)
                self.score += self.data[self.day_index] - bought_price
                #print ("Sell: " + formatPrice(self.data[self.day_index]) + " | Profit: " + formatPrice(self.data[self.day_index] - bought_price))
                
                self.actions[0].append( self.day_index )
                self.actions[1].append( self.data[self.day_index] )
                self.act_type.append(0)
            
        elif fm[1] == 1: # stay
            pass
        
        
        elif fm[2] == 1: # buy
            self.inventory.append(self.data[self.day_index])
            #print( "Buy: " + formatPrice(self.data[self.day_index]))
            
            self.actions[0].append( self.day_index )
            self.actions[1].append( self.data[self.day_index] )
            self.act_type.append(1)
        
        done = True if self.day_index == len(self.data) - 1 else False
        
        if done:
            print ("--------------------------------")
            print ("Total Profit: " + formatPrice(self.score))
            print ("--------------------------------")
        
        self.score += reward

        self.day_index += 1
        
        return reward, self.score, done

import random
from collections import deque
import torch
import numpy as np
#from detraffic.model import Linear_QNet, QTrainer


MAX_MEMORY = 1_000_000
BATCH_SIZE = 64
LR = 0.001


class Agent:
    def __init__(self, epsilon=120, gamma=0.9):
        self.n_games = 0
        self.epsilon = epsilon  # randomness
        self.gamma = gamma  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(8, [64, 128, 256], 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        self.w_size = 7


    def get_state(self, game: StockData, t):
        #state = game.data[0]
        
        state = getState(game.data, t, self.w_size + 1)
        
        minStock = game.inventory[0] if len(game.inventory) > 0 else 0
        state = np.append(state, minStock)
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append(
            (state, action, reward, next_state, game_over)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        # for state, action, reward, nexrt_state, game_over in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # [ sell, stay, buy]
        final_move = [0, 0, 0]
        if random.randint(1, 200) <= self.epsilon:
            move = random.randint(0, len(final_move) - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
        
            
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # print(final_move)

        return final_move


import Helper

def train(data):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = -math.inf
    agent = Agent()
    game = StockData(data)
    
    index = 0
    while True:
        # get old state
        #state = getState(self.data, 0, self.w_size + 1)
        state_old = agent.get_state(game,game.day_index)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, score, game_over = game.play_step(final_move)
        
        #next_state = getState(self.data, t + 1, self.w_size + 1)
        state_new = agent.get_state(game,game.day_index)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)
        
        if game_over or game.day_index >= len(game.data) - 1:
            
            #f_score = sum([(game.data[-1] - i) for i in game.inventory]) + score
            
            print ("--------------------------------")
            print ("fs: " + str(score))
            print ("record: " + str(record))
            print ("Total Profit: " + formatPrice(score))
            print ("--------------------------------")
            
            save = False
            
            # train long memory, plot result
            
            agent.n_games += 1
            agent.epsilon = max(0, agent.epsilon - 1)
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("CEF_epc_" + str(index))
                save = True
                
                index += 1

            # print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            Helper.plot_custom(game.data, game.actions, game.act_type, save, index)
            
            Helper.plot_progress(plot_scores, plot_mean_scores, save, index)
            game.reset()

train(getStockDataVec(r"CEF"))