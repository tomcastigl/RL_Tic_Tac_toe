# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:02:02 2022

@author: fadel
"""
import numpy as np
from tic_env import TictactoeEnv,OptimalPlayer
import pickle
import matplotlib.pyplot as plt
from utils import agent,test_policy,update_player,get_hash,get_action


def q_learning(epsilon,num_episodes:int,env:TictactoeEnv,path_save:str,eps_opt=0.5,alpha=0.05,gamma=0.99,render=False,test=False):
    
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
    
    for episode in range(1,num_episodes+1):
        
        if episode % 250 == 0 :
            agent_mean_rewards[int(episode//250)-1] = accumulate_reward/250
            accumulate_reward = 0 # reset

            if test:
                M_opt = test_policy(0,q_table=q_table,verbose=False)
                M_rand = test_policy(1,q_table=q_table,verbose=False)
                M_opts.append(M_opt)
                M_rands.append(M_rand)
                    
        env.reset()
        #turns = turns[np.random.permutation(2)] # alternating 1st player 
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
        players[turns[1]]='agent'
        
        current_state = None
        next_state = None
        A = None
        
        while not env.end : # The game takes at most 9 steps to finish

            #-- Optimal player plays 
            if env.current_player == turns[0]  :
                grid,end,_ = env.observe() #-- observe grid
                move = player_opt.act(grid) #-- get move
                env.step(move,print_grid=False) # optimal player takes a move
                
                #-- update agent q_table after optimal player takes a step
                if current_state is not None :
                    reward = env.reward(agent_learner)
                    q_table = update_player(q_table,current_state,env.observe()[0],A,alpha,reward,gamma) # the next state where agent can play is after optimal has played

            #-- agent plays
            else:   
                
                #-- Learning agent updates q_table
                current_state = env.observe()[0]
                current_state_hash = get_hash(current_state) 
                
                # Add current_state in q_table if applicable
                if not(current_state_hash in q_table.keys()): 
                    q_table[current_state_hash]=np.zeros((1,9))
                            
                #----- Choose action A with epsilon greedy
                A = get_action(epsilon(episode), current_state, current_state_hash,q_table)
                
                #----- Take action A & Observe reward
                next_state,end,_ = env.step(A,print_grid=False)
                reward = env.reward(agent_learner)
                
                if reward > 0:  
                    #----- Update q_table when agent wins
                    q_table = update_player(q_table,current_state,None,A,alpha,reward,gamma)
                    

            #-- Chek that the game hasn't finished
            if env.end :
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
                    
            
        if episode % 1000 == 0 :
            print(f"\nEpisode : {episode}")
            print(wins_count)
            
    #--save
    if path_save is not None:
        with open(path_save,'wb') as file:
            pickle.dump(q_table, file)
    
    return q_table,wins_count,agent_mean_rewards,M_opts,M_rands


#-- Q.1
eps_1=lambda x : 0.1
path = './20k_eps_01-v2.pickle'
if True:
    env = TictactoeEnv()
    q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_1,num_episodes=int(20e3),eps_opt=0.5,env=env,path_save=path,alpha=0.05,gamma=0.99,render=False,test=False)
    plt.plot(agent_mean_rewards)
    plt.title(f'Average rewards with Epsilon {eps_1(0)}')

#-- Q.2 and Q.3
eps_min=0.1
eps_max=0.8
if False :
    fig,axs = plt.subplots(3,1,figsize=(10,10))
    env = TictactoeEnv()
    test=True
    for N_star in [1,10e3,20e3,30e3,40e3]:
        print('--'*20,'>',N_star)
        eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
        q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_2,num_episodes=int(20e3),env=env,path_save=f'./20k_decreasing_Nstar{N_star}.pkl',alpha=0.05,gamma=0.99,render=False,test=test)
        axs[0].plot(agent_mean_rewards,label=f'N* {N_star}')
        axs[1].plot(M_opts,label=f' M_opt N* {N_star}')
        axs[2].plot(M_rands,label=f' M_rand N* {N_star}')
    #print('\nfinal : ',wins_count)
    axs[0].set_title('Average rewards')
    axs[1].set_title(r'$M_{opt}$')
    axs[2].set_title(r'$M_{rand}$')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.tight_layout()
    
"""
#-- Plot epsilons profiles
for N_star in [1,10e3,20e3,30e3,40e3]:
    eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
    plt.plot(list(map(eps_2,range(1,20000))),label=f'N_star {N_star}')
plt.title("Epsilons")
plt.legend() 
"""
#-- Q.4
if False :
    env = TictactoeEnv()
    test=True
    N_star = 10e3 # to update
    Mopts=list()
    Mrands=list()
    for eps in [0,0.25,0.5,0.75,1]:
        eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
        q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_2,eps_opt=eps,num_episodes=int(20e3),env=env,path_save=None,alpha=0.05,gamma=0.99,render=False,test=test)
        plt.plot(M_opts,label=f'M_opt; eps_opt={eps}')
        plt.plot(M_rands,label=f'M_rand; eps_opt={eps}')
        
        Mopts.append(M_opts) # for debugging 
        Mrands.append(M_rands) # for debugging
#-- Test
for i in range(10):
    print(i,"--"*20)
if False :
    for i in range(10):
        print(i,'--'*20)
        env = TictactoeEnv()
        q_table = None
        with open(path,'rb') as file:
            q_table = pickle.load(file)
        
        print('path : ', path)
        player = agent(q_table)
        for eps in [0,1]:
            test_policy(eps,q_table,verbose=True)








