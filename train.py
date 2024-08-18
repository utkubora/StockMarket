import keras


from Helper import *
import sys

class ModelTrainer:
    
    def __init__(self,agent, data, w_size, epoch, df_name) -> None:
        #agent = Agent(window_size)
        #data = getStockDataVec(stock_name)
        
        self.data = data
        self.agent = agent
        
        self.w_size = w_size
        self.epoch = epoch
        self.df_name = df_name
        
        
    def fit(self, batch_size = 1000):
        
        l = len(self.data) - 1
        scores = []
        mean_scores = []
        
        for e in range(self.epoch + 1):
            actions = []
            print ("Episode " + str(e) + "/" + str(self.epoch))
            
            state = getState(self.data, 0, self.w_size + 1)

            total_profit = 0
            self.agent.inventory = []

            for t in range(l):
                action = self.agent.act(state)
                actions.append(action)

                # sit
                next_state = getState(self.data, t + 1, self.w_size + 1)
                reward = 0

                if action == 1: # buy
                    self.agent.inventory.append(self.data[t])
                    print( "Buy: " + formatPrice(self.data[t]))

                elif action == 2 and len(self.agent.inventory) > 0: # sell
                    bought_price = self.agent.inventory.pop(0)
                    reward = max(self.data[t] - bought_price, 0)
                    total_profit += self.data[t] - bought_price
                    print ("Sell: " + formatPrice(self.data[t]) + " | Profit: " + formatPrice(self.data[t] - bought_price))

                done = True if t == l - 1 else False
                self.agent.memory.append((state, action, reward, next_state, done))
                scores.append(total_profit)
                mean_scores.append( sum(scores) / len(scores) )
                state = next_state

                if done:
                    print ("--------------------------------")
                    print ("Total Profit: " + formatPrice(total_profit))
                    print ("--------------------------------")

                if len(self.agent.memory) > batch_size:
                    self.agent.expReplay(batch_size)
                
            actions.append(action)
            plot_progress(scores,mean_scores=mean_scores,save=True,index=str(e))
            plot_custom(self.data,actions,actions,True,e)

            if e % 10 == 0:
                self.agent.model.save("models/model_" + self.df_name + "_epF" + str(e) + ".h5")
                
                
                
