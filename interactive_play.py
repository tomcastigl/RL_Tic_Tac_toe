
import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import torch
import torch.nn as nn
import random
import time

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()

        self.fc=nn.Sequential(
            nn.Linear(3*3*2,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,9)
        )
    
    def forward(self,x):
        x=x.view(-1,18)
        x=self.fc(x)
        return x   

def grid2tensor(grid,player):
    tens=torch.zeros(3,3,2)
    grid=torch.from_numpy(grid)
    if player=='X':
        player_index=torch.nonzero(grid==1,as_tuple=False)
        op_index=torch.nonzero(grid==-1,as_tuple=False)
    if player=='O':
        player_index=torch.nonzero(grid==-1,as_tuple=False)
        op_index=torch.nonzero(grid==1,as_tuple=False)
    tens[player_index[:,0],player_index[:,1],0]=1
    tens[op_index[:,0],op_index[:,1],1]=1
    return tens

class human_player:
    def __init__(self, player='X'):
        self.player=player

class deep_Q_player:
    def __init__(self,player='X', epsilon=.1):
        self.player=player
        self.epsilon=epsilon
    
        
    def act(self,state,model):
        
        action = model(state).max(1)[1].view(-1,1)
        if np.random.random() < self.epsilon:
            action = torch.tensor([[random.randrange(9)]])
        
        return action

start=input('commencer? (oui/non): ')
env=TictactoeEnv()
if start=='oui':
    player=human_player('X')
    opponent=deep_Q_player('O', epsilon=0)
if start=='non':
    player=human_player('O')
    opponent=deep_Q_player('X', epsilon=0)
policy=DQN()
policy.load_state_dict(torch.load('DQN_best.pkl'))
i=0

while not env.end:
    
    if i==0: env.render()    
    if env.current_player==player.player:
        move=int(input('enter your move [1,9]:'))-1
        env.step(move)
        env.render()
    elif env.current_player==opponent.player:
        time.sleep(1)
    
        state=grid2tensor(env.grid,opponent.player)
        move=int(opponent.act(state,policy))
        env.step(move)
        env.render()
    
    if env.end:
        if env.winner==opponent.player:
            print('SUCK IT! \n\n' )
            
        elif env.winner==player.player:
            print('YOU BEAT ME! \n\n')
        else:
            print('ITS A TIE! \n\n')
    i+=1

        





