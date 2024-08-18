from train import ModelTrainer
import sys

from Agent.agent import Agent
from Helper import *

if __name__ == '__main__':
    
    if len(sys.argv) != 5: 
        print('Error: Parameter sizes not right. Expected usage: "python main.py <prm1> <prm2> <prm3> <prm4>" ')
    
    else:
        prm1 = sys.argv[1]
        prm2 = sys.argv[2]
        prm3 = sys.argv[3]
        prm4 = sys.argv[4]
        
        if prm1 == "train":
            #train runned
            data = getStockDataVec(prm3)
            w_size = int(prm2)
            epoch = int(prm4)
            agent = Agent(w_size)
            
            trainer = ModelTrainer(
                agent=agent,
                data=data,
                w_size=w_size,
                epoch=epoch,
                df_name=prm3
            )
            trainer.fit()
        
        elif prm1 == "evaluate":
            #eval runned
            pass