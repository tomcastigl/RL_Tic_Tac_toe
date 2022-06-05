# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:02:02 2022

@author: fadel
"""
import numpy as np
from tic_env import TictactoeEnv,OptimalPlayer
import pickle
from utils import test_policy,update_player,get_hash,get_action
import wandb

def q_learning(epsilon,num_episodes:int,env:TictactoeEnv,path_save:str,eps_opt=0.5,alpha=0.05,gamma=0.99,render=False,test=False,tag='',wand_name='Tabular-Q'):
    
    q_table = dict()
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
    
    #-- Init
    wandb.init(tags=[tag],project='ANN',entity='fadelmamar', 
               name=wand_name, config={'alpha':alpha,'gamma':gamma,'eps_opt':eps_opt})
    
    for episode in range(1,num_episodes+1):
        
        if episode % 250 == 0 :
            agent_mean_rewards[int(episode//250)-1] = accumulate_reward/250
            wandb.log({'average_reward':accumulate_reward/250})
            accumulate_reward = 0 # reset

            if test:
                M_opt = test_policy(0,q_table=q_table,verbose=False)
                M_rand = test_policy(1,q_table=q_table,verbose=False)
                M_opts.append(M_opt)
                M_rands.append(M_rand)
                wandb.log({'M_opt':M_opt,'M_rand':M_rand})
                    
        env.reset()     
        #-- Permuting player every 2 games
        if episode % 2 == 0:
            turns[0] = 'X'
            turns[1] = 'O'
        else:
            turns[0] = 'O'
            turns[1] = 'X'
        
        player_opt = OptimalPlayer(epsilon=eps_opt,player=turns[0])
        agent_learner = turns[1]       
        players[turns[0]]='optimal_player'
        players[agent_learner]='agent'
        
        current_state = None
        A = None
        
        for j in range(9) : # The game takes at most 9 steps to finish
            
            #-- Agent plays
            if env.current_player == agent_learner :                 
                #-- Learning agent updates q_table
                current_state = env.observe()[0]
                current_state_hash = get_hash(current_state) 
                
                # Add current_state in q_table if needed
                if not(current_state_hash in q_table.keys()): 
                    q_table[current_state_hash]=np.zeros((1,9))                           
                
                A = get_action(epsilon(episode), current_state,q_table) #-- Choose action A with epsilon greedy
                _,_,_ = env.step(A,print_grid=False) #-- Take action A 
                
            #-- Optimal player plays 
            if not env.end :
                grid,end,_ = env.observe() #-- observe grid
                move = player_opt.act(grid) #-- get move
                env.step(move,print_grid=False) # optimal player takes a move
                
                #-- update agent q_table after optimal player takes a step
                if current_state is not None :
                    agent_reward = env.reward(agent_learner)
                    next_state = env.observe()[0]
                    q_table = update_player(q_table,current_state,next_state,A,alpha,agent_reward,gamma) 
              
            #-- Chek that the game has finished
            if env.end :
                
                agent_reward = env.reward(agent_learner)
                q_table = update_player(q_table,current_state,None,A,alpha,agent_reward,gamma) #-- Update q_table when game ends
                
                if env.winner is None :
                    wins_count['draw'] = wins_count['draw'] + 1   
                else :
                    winner = players[env.winner]
                    wins_count[winner] = wins_count[winner] + 1
                    
                if render : 
                    print(f"Episode {episode} ; Winner is {winner}.")
                    env.render()

                #-- accumulate rewards
                accumulate_reward += env.reward(agent_learner)
                
                break
                    
            
        if episode % 1000 == 0 :
            print(f"\nEpisode : {episode}")
            print(wins_count)
            
    #--save
    if path_save is not None:
        with open(path_save,'wb') as file:
            pickle.dump(q_table, file)
            
    wandb.finish()
    
    return q_table,wins_count,agent_mean_rewards,M_opts,M_rands


#-- Q.1
eps_1=lambda x : 0.3
if True:
    env = TictactoeEnv()
    q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_1,num_episodes=int(20e3),eps_opt=0.5,
                                                                      env=env,path_save=None,alpha=0.05,tag='Q.1',
                                                                      gamma=0.99,render=False,test=False,wand_name='Tabular-Q')

#-- Q.2 and Q.3
eps_min=0.1
eps_max=0.8
if False :
    env = TictactoeEnv()
    test=True
    for N_star in [1,10e3,20e3,30e3,40e3]:
        print('--'*20,'>',N_star)
        eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
        q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_2,num_episodes=int(20e3),env=env,tag='Q.2-Q.3',
                                                                          wand_name=f'TabularQ--Nstar-{N_star}',
                                                                          path_save=None,alpha=0.05,gamma=0.99,render=False,test=test)

#-- Q.4
if False :
    env = TictactoeEnv()
    test=True
    N_star = 10e3 # to update
    Mopts=list()
    Mrands=list()
    for eps in [0,0.25,0.5,0.75,1]:
        eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
        q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_2,eps_opt=eps,num_episodes=int(20e3),tag='Q.4--Nstar-{N_star}',wand_name=f'TabularQ--eps_opt-{eps}',
                                                                          env=env,path_save=None,alpha=0.05,gamma=0.99,render=False,test=test)
        





