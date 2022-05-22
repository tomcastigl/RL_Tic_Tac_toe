# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:42:06 2022

@author: fadel
"""

import numpy as np
from tic_env import TictactoeEnv,OptimalPlayer
from utils import test_policy,grid2tensor,DQN,ReplayMemory,deep_Q_player
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from typing import Sequence


def update_policy(policy_net:nn.Module,
                  target_net:nn.Module,
                  memory:ReplayMemory,
                  optimizer:optim,
                  criterion=F.huber_loss,
                  gamma=0.99,
                  online_update=False,online_state_action_reward=(None,None,None,None)):
    
    memory.step += 1
    loss = None
    
    if online_update :
        #assert None not in online_state_action_reward,'provide these values.'
        
        #-- Compute Q values
        state,next_state,action,reward = online_state_action_reward
        state=state.to(memory.device)
        state_action_values = policy_net(state)[:,action] # take Q(state,action)
        
        next_state_values=torch.tensor([0.0])
        if next_state is not None:
            next_state = next_state.to(memory.device)
            next_state_values = target_net(next_state).max(1)[0].detach() # take max Q(state',action')
        
        #-- Compute target
        target = reward + gamma*next_state_values # no terminal state
        
        #-- Update gradients
        loss = criterion(state_action_values,target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
        
        #-- Log
        wandb.log({'loss':loss.item(),'reward':reward,'Step':memory.step})

    else:
        if len(memory) < memory.batch_size:
            return
        
        #-- Sample Transitions
        transitions = memory.sample()
        
        #-- GetTransition of batch-arrays
        batch = memory.Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=memory.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state).to(memory.device)
        action_batch = torch.cat(batch.action).to(memory.device)
        reward_batch = torch.cat(batch.reward).to(memory.device)
        
        #-- Compute Q values
        state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        #-- Compute target
        next_state_values = torch.zeros(memory.batch_size,device=memory.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        target = reward_batch + gamma*next_state_values
        
        #-- Update gradients
        loss = criterion(state_action_values,target.unsqueeze(1),reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
        
        #-- Log
        wandb.log({'loss':loss,'mean_batch_reward':reward_batch.float().mean(),'Step':memory.step})
    
    memory.accumulated_loss += loss.item()
    return loss.item()


def deep_q_learning(epsilon,num_episodes:int,
                    env:TictactoeEnv,
                    path_save:str,
                    eps_opt=0.5,gamma=0.99,
                    render=False,test=False,online_update=False,wandb_tag="DQN"):
    #-- agent
    agent = deep_Q_player()
    
    #-- Initialize Q network
    policy_net = DQN()
    
    #-- Initialize target network
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    batch_size=64
    gamma=0.99
    lr=5e-4
    optimizer = optim.Adam(policy_net.parameters(),lr=lr)
    memory = ReplayMemory(10000,batch_size)
    args={'gamma':gamma,'batch_size':batch_size,'replay_buffer':int(1e4),'lr':lr,'eps_opt':eps_opt,'online_update':online_update}
    
    #comouting device
    #if torch.cuda.is_available():
       # memory.device.to('cuda:0')
        
    policy_net.to(memory.device)
    target_net.to(memory.device)
    
    #-- wandb init
    wandb.init(tags=[wandb_tag],
               project='ANN', 
               entity='fadelmamar', 
               name=str('DQN_learnFromXpert-'+wandb_tag), 
               config=args)
    turns = np.array(['X','O'])
    
    #-- Holder 
    wins_count = dict()
    wins_count['optimal_player']=0
    wins_count['agent']=0
    wins_count['draw']=0
    players = dict()
    M_opts = list()
    M_rands = list()
    accumulate_reward = 0
    agent_mean_rewards = [0]*int(num_episodes//250)
    num_illegal_actions = 0
    
    for episode in range(1,num_episodes+1):
        
        wandb.log({'episode':episode,'epsilon_greedy':epsilon(episode)})
        
        if episode % 250 == 0 :
            agent_mean_rewards[int(episode//250)-1] = accumulate_reward/250
            wandb.log({'mean_reward':accumulate_reward/250,'mean_loss':memory.accumulated_loss/250})
            accumulate_reward = 0 # reset
            memory.accumulated_loss = 0 # reset
            
            if test:
                M_opt = test_policy(0,q_table=None,DQN_policy_net=policy_net,verbose=False)
                M_rand = test_policy(1,q_table=None,DQN_policy_net=policy_net,verbose=False)
                M_opts.append(M_opt)
                M_rands.append(M_rand)
                wandb.log({'M_opt':M_opt,'M_rand':M_rand})
            
        #-- Upddate target network
        if episode % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        env.reset()
        turns = turns[np.random.permutation(2)] # alternating 1st player 
        player_opt = OptimalPlayer(epsilon=eps_opt,player=turns[0])
        agent_learner = turns[1]
        
        players[turns[0]]='optimal_player'
        players[turns[1]]='agent'
        
        current_state = None
        A = None # action
        
        while not env.end : 

            #-- Optimal player plays 
            if env.current_player == turns[0] :
                grid,end,_ = env.observe() #-- observe grid
                move = player_opt.act(grid) #-- get move
                env.step(move,print_grid=False) # optimal player takes a move
                
                #-- Update agent after optimal player takes a step
                if current_state is not None :
                    next_state = grid2tensor(env.observe()[0],agent_learner)
                    reward = env.reward(agent_learner)
                    
                    if reward < 0 : 
                        next_state = None # optimal wins
                    
                    #-- Update Policy
                    update_policy(policy_net,target_net, memory,
                                      optimizer, gamma=gamma,
                                      online_update=online_update,
                                      online_state_action_reward=(current_state,next_state,A,reward))
                    
                        
                    # push in replay buffer
                    if not online_update:
                        memory.push(current_state, torch.tensor([A]), next_state, torch.tensor([reward])) 

            #-- agent plays
            else:   
                
                current_state = grid2tensor(env.observe()[0],agent_learner)
                            
                #----- Choose action A with epsilon greedy
                A = agent.act(current_state, policy_net,epsilon(episode))  
                wandb.log({'action':A})

                #----- Take action A & Observe reward
                try :
                    _,_,_ = env.step(A,print_grid=False)
                    reward = env.reward(agent_learner)

                    #----- Update when agent wins
                    if env.reward(agent_learner) > 0:  
                        reward = 1
                        next_state = None
                        
                        if online_update:
                            update_policy(policy_net,target_net, memory,
                                          optimizer, gamma=gamma,
                                          online_update=True,
                                          online_state_action_reward=(current_state,next_state,A,reward))
                            
                        else :
                            memory.push(current_state, torch.tensor([A]), next_state, torch.tensor([reward])) # update replay buffer

                        
                #----- Update when agent moves illegaly
                except ValueError :
                    reward = -1.0
                    next_state = None
                    
                    if online_update:
                        update_policy(policy_net,target_net, memory,
                                      optimizer, gamma=gamma,
                                     online_update=True,
                                     online_state_action_reward=(current_state,next_state,A,reward))
                        
                    else:
                        memory.push(current_state, torch.tensor([A]), next_state, torch.tensor([reward])) # update replay buffer

                    num_illegal_actions += 1
                    wandb.log({'num_illegal_actions':num_illegal_actions})
                    
                    #-- Terminating game
                    env.end = True
                    env.winner = turns[0] # optimal player
                    
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
                
                
        if episode % 1000 == 0 :
            print(f"\nEpisode : {episode}")
            print(wins_count)
                
    wandb.finish()
    
    
    return wins_count,agent_mean_rewards,M_opts,M_rands

#-- Q.11 & Q.12
eps_1=lambda x : 0.3
if True:
    test=False
    for do in [False,True]:
        env = TictactoeEnv()
        wins_count,agent_mean_rewards,M_opts,M_rands = deep_q_learning(epsilon=eps_1,num_episodes=int(20e3),
                                                                          eps_opt=0.5,env=env,path_save=None,
                                                                          gamma=0.99,render=False,test=test,
                                                                          wandb_tag="V2",online_update=do)
    
#-- Q.13
eps_min=0.1
eps_max=0.8
if True :
    env = TictactoeEnv()
    test=True
    for do in [True,False]:
        for N_star in [1,10e3,20e3,30e3,40e3]:
            print('--'*20,' N_star : ',N_star)
            eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
            wins_count,agent_mean_rewards,M_opts,M_rands = deep_q_learning(epsilon=eps_2,num_episodes=int(20e3),
                                                                          eps_opt=0.5,env=env,path_save=None,
                                                                          gamma=0.99,render=False,test=test,
                                                                          wandb_tag="V2",online_update=do)
        
"""
env = TictactoeEnv()
env.step(1)
env.step(2)
env.step(3)
grid = env.observe()[0]

tens = grid2tensor(grid,'X')
print(tens.sum(2))
print(grid)
#print(tens.shape)
#print(tens[:,:,0])
#print(tens[:,:,1])
"""








