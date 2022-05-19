# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:42:06 2022

@author: fadel
"""

import numpy as np
from tic_env import TictactoeEnv,OptimalPlayer
import pickle
import matplotlib.pyplot as plt
from utils import test_policy,update_player,get_hash,get_action
import matplotlib
from collections import namedtuple, deque
from itertools import count
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from utils import test_policy

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()

        self.fc=nn.Sequential(nn.Linear(3*3*2,128),nn.ReLU(),
                              nn.Linear(128,128),nn.ReLU(),
                              nn.Linear(128,128),nn.ReLU(),
                              nn.Linear(128,9))
    
    def forward(self,x):
        x=x.view(-1,18)
        x=self.fc(x)
        return x

class ReplayMemory(object):

    def __init__(self, capacity,batch_size):
        self.memory = deque([],maxlen=capacity)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.device='cpu'
        self.batch_size=batch_size
        self.step = 0
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

class deep_Q_player:
    def __init__(self,player='X'):
        self.player=player
    
        
    def act(self,state,model:nn.Module,epsilon:float):
        
        with torch.no_grad():
            action_scores = model(state)
        
        action = action_scores.argmax().item()
        if np.random.random() < epsilon:
            action = np.random.choice(range(9)).item()
        
        return action

def grid2tensor(grid:np.array,player:str):
    
    state=np.zeros((3,3,2))
    a = 2*(player=='X')-1 #  1 if player='X' and -1 otherwise
    
    grid1 = np.where(grid==a,1,0)
    grid2 = np.where(grid==-a,1,0)
    
    state[:,:,0]=grid1
    state[:,:,1]=grid2
    
    return torch.tensor(state).float()
def update_policy(policy_net:nn.Module,
                  target_net:nn.Module,
                  memory:ReplayMemory,
                  optimizer:optim,
                  criterion=F.huber_loss,
                  gamma=0.99,terminal_state=False,terminal_state_action_reward=(None,None,None)):
    
    memory.step += 1
    
    if terminal_state :
        assert None not in terminal_state_action_reward,'provide these values.'
        
        #-- Compute Q values
        state,action,reward = terminal_state_action_reward
        state=state.to(memory.device)
        state_action_values = policy_net(state)[0,action] # take Q(state,action)
        
        #-- Compute target
        target = torch.tensor(reward + gamma*0.0,device=memory.device) # no terminal state
        
        #-- Update gradients
        #criterion = nn.SmoothL1Loss()

        loss = criterion(state_action_values,target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
        
        #-- Log
        wandb.log({'loss':loss.item(),'reward':reward,'action':action,'Step':memory.step})
        
        return loss.item()
       
    
    if len(memory) < memory.batch_size:
        return
    
    #-- Sample Transitions
    transitions = memory.sample()
    
    #-- GetTransition of batch-arrays
    batch = memory.Transition(*zip(*transitions))
    
    non_final_next_states = torch.cat([s for s in batch.next_state]) # if s is not None])
    
    state_batch = torch.cat(batch.state).to(memory.device)
    action_batch = torch.cat(batch.action).to(memory.device)
    reward_batch = torch.cat(batch.reward).to(memory.device)
    
    #-- Compute Q values
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
    #-- Compute target

    next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    target = reward_batch + gamma*next_state_values
    
    #-- Update gradients
    loss = criterion(state_action_values,target.unsqueeze(1),reduction='mean')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step
    
    #-- Log
    wandb.log({'loss':loss,'mean_batch_reward':reward_batch.float().mean(),'Step':memory.step})

    return loss.item()



def deep_q_learning(epsilon,num_episodes:int,
                    env:TictactoeEnv,
                    path_save:str,
                    eps_opt=0.5,gamma=0.99,
                    render=False,test=False):
    #-- agent
    agent = deep_Q_player()
    agent.epsilon=0.0
    
    #-- Initialize Q network
    policy_net = DQN()
    
    #-- Initialize target network
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    batch_size=64
    gamma=0.99
    optimizer = optim.Adam(policy_net.parameters(),lr=5e-4)
    memory = ReplayMemory(10000,batch_size)
    args={'gamma':gamma,'batch_size':batch_size,'replay_buffer':int(1e4),'lr':5e-4,'eps_opt':eps_opt}
    
    policy_net.to(memory.device)
    target_net.to(memory.device)
    
    #-- wandb init
    wandb.init(tags='DQN-2',
               project='ANN', 
               entity='fadelmamar', 
               name='DQN_learnFromXpert', 
               config=args)
    
    turns = np.array(['X','O'])
    
    #-- Holder 
    wins_count = dict()
    wins_count['optimal_player']=0
    wins_count['agent']=0
    wins_count['draw']=0
    players = dict()
    players[None] = 'draw'
    M_opts = list()
    M_rands = list()
    accumulate_reward = 0
    agent_mean_rewards = [0]*int(num_episodes//250)
    num_illegal_actions = 0
    
    for episode in range(1,num_episodes+1):
        wandb.log({'episode':episode,'epsilon_greedy':epsilon(episode)})
        if episode % 250 == 0 :
            agent_mean_rewards[int(episode//250)-1] = accumulate_reward/250
            wandb.log({'mean_reward':accumulate_reward/250})
            accumulate_reward = 0 # reset
            
            if test:
                #M_opt = test_policy(0,q_table,verbose=False)
                #M_rand = test_policy(1,q_table,verbose=False)
                #M_opts.append(M_opt)
                #M_rands.append(M_rand)
                pass
            
        #-- Upddate target network
        if episode % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % 1000 == 0 :
                print(f"\nEpisode : {episode}")
                print(wins_count)
        
        env.reset()
        turns = turns[np.random.permutation(2)] # alternating 1st player 
        player_opt = OptimalPlayer(epsilon=eps_opt,player=turns[0])
        agent_learner = turns[1]
        agent.player = agent_learner
        
        players[turns[0]]='optimal_player'
        players[turns[1]]='agent'
        
        current_state = None
        A = None # action
        for j in range(9) : # The game takes at most 9 steps to finish

            #-- Optimal player plays 
            if env.current_player == turns[0] :
                grid,end,_ = env.observe() #-- observe grid
                move = player_opt.act(grid) #-- get move
                env.step(move,print_grid=False) # optimal player takes a move
                
                #-- update agent after optimal player takes a step
                if current_state is not None :
                    #next_state = torch.from_numpy(env.observe()[0])
                    next_state = grid2tensor(env.observe()[0],agent_learner)
                    reward = env.reward(agent_learner)
                    
                    if reward < 0 : # optimal wins
                        update_policy(policy_net,target_net, memory,
                                      optimizer,gamma=gamma,
                                      terminal_state=True,terminal_state_action_reward=(current_state,A,reward))
                        
                    else: # no winner yet
                        update_policy(policy_net,target_net, memory,
                                      optimizer, gamma=gamma,
                                      terminal_state=False,terminal_state_action_reward=(None,None,None))
                    
                    # push in replay buffer
                    memory.push(current_state, torch.tensor([A]), next_state, torch.tensor([reward])) 

            #-- agent plays
            else:   
                
                current_state = grid2tensor(env.observe()[0],agent_learner) #-- Get state
                            
                #----- Choose action A with epsilon greedy
                A = agent.act(current_state, policy_net,epsilon(episode))  
                
                #----- Take action A & Observe reward
                try :
                    _,_,_ = env.step(A,print_grid=False)
                    reward = env.reward(agent_learner)
                    
                #----- Update when agent moves illegaly
                except ValueError :
                    reward = -1.0
                    update_policy(policy_net,target_net, memory,
                                  optimizer, gamma=gamma,
                                  terminal_state=True,terminal_state_action_reward=(current_state,A,reward))
                    #env.end = True
                    #env.winner = turns[0] # optimal player wins
                    num_illegal_actions += 1
                    wandb.log({'num_illegal_actions':num_illegal_actions})
                    
                    break # terminates game
                
                #----- Update when agent wins
                if reward > 0:  
                    update_policy(policy_net,target_net, memory,
                                  optimizer, gamma=gamma,
                                  terminal_state=True,terminal_state_action_reward=(current_state,A,reward))
                    
            #-- Chek that the game hasn't finished
            if env.end :
                if env.winner is not None :
                    winner = players[env.winner]
                    wins_count[winner] = wins_count[winner] + 1
                else :
                    wins_count['draw'] = wins_count['draw'] + 1

                if render : 
                    print(f"Episode {episode} ; Winner is {winner}.")
                    env.render()
                    
                #-- accumulate rewards
                accumulate_reward += env.reward(agent_learner)
                
                wandb.log({'accumulated_reward':accumulate_reward,
                           'reward':reward})
                wandb.log(wins_count)
                    
                break
    
    wandb.finish()
    
    
    return wins_count,agent_mean_rewards,M_opts,M_rands


eps_1=lambda x : 0.1
if True:
    env = TictactoeEnv()
    wins_count,agent_mean_rewards,M_opts,M_rands = deep_q_learning(epsilon=eps_1,num_episodes=int(20e3),
                                                                   eps_opt=0.5,env=env,path_save=None,
                                                                   gamma=0.99,render=False,test=False)
    plt.plot(agent_mean_rewards)
    plt.title(f'Average rewards with Epsilon {eps_1(0)}')

"""
env = TictactoeEnv()
env.step(1)
env.step(2)
env.step(3)
grid = env.observe()[0]

tens = grid2tensor(grid,'X')
print(grid)
print(tens.shape)
print(tens[:,:,0])
print(tens[:,:,1])
"""








