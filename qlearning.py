# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:02:02 2022

@author: fadel
"""
import numpy as np
from tic_env import TictactoeEnv,OptimalPlayer
import pickle
import matplotlib.pyplot as plt
#################################################################################
#-------------------------------------------- Helpers
def get_hash(state:np.array)->str:
    
    return str(state.flatten(order='C'))


def get_available_positions(state:np.array):
      
    flatten_array = state.flatten(order='C')
    indices = np.argwhere(np.abs(flatten_array)<1).T
    
    return indices.flatten()

def get_best_action(availables_positions:np.array,array_of_q_values:np.array):
    
    A = np.argmax(array_of_q_values,axis=1)[0]
    #print(availables_positions)
    
    if  A in availables_positions :
        return A.item()
    
    else :
        A = availables_positions[0]
        for i in availables_positions :
            if array_of_q_values[:,i]>array_of_q_values[:,A]:
                A = i
        return A.item()

def get_action(epsilon,current_state,current_state_hash,q_table):
    
    availables_positions = get_available_positions(current_state)
    rnd = np.random.uniform(0,1)
                
    if rnd > epsilon :
        A = get_best_action(availables_positions,q_table[current_state_hash]) # best move among possible moves        
    else:
        np.random.shuffle(availables_positions) # shuffle
        A = availables_positions[0] # take first
        A = A.item()
    
    return A
    

def get_max_Q(q_table:dict,state:np.array):
        
    hash_ = get_hash(state)
    max_q = 0.
    
    if hash_ in q_table.keys():
        max_q = np.max(q_table[hash_],axis=1)
        
    return max_q
    
#---------------------------------------------------
#################################################################################
def update_player(q_table,current_state,next_state,A,alpha,reward,gamma):
    
    current_state_hash = get_hash(current_state)
    
    #----- Update q_table
    array_of_q_values = q_table[current_state_hash]
    q_value = array_of_q_values[:,A]
    updated_q_value = q_value + alpha*(reward + gamma*get_max_Q(q_table,next_state) - q_value)
    array_of_q_values[:,A] = updated_q_value # update Q(S,A)
    q_table[current_state_hash] = array_of_q_values # store updated Q(S,A)
    
    return q_table


class agent:
    
    def __init__(self,q_table):
        self.q_table = q_table
        
    def act(self,grid):
        
        grid_hash = get_hash(grid)
        availables_positions = get_available_positions(grid)
        
        #-- for states in the q_table
        if grid_hash in q_table.keys():
            A = get_best_action(availables_positions,q_table[grid_hash])
        
        #-- For any other state not discovered, any available action can be taken
        else :
            np.random.shuffle(availables_positions)
            A = availables_positions[0].item()
            
        return A

def test_policy(eps_optimalplayer,q_table,verbose=False):
    
    env = TictactoeEnv() # environment
    agent_player = agent(q_table) # agent
    
    #-- Holders 
    wins_count = dict()
    wins_count['optimal_player']=0
    wins_count['agent']=0
    wins_count['draw']=0 
    players = dict()
    players[None] = 'draw'
    turns = np.array(['X','O'])
    
    for episode in range(500):
        
        env.reset()
        np.random.seed(episode) 
        
        if episode < 250 :
            player_opt = OptimalPlayer(epsilon=eps_optimalplayer,player=turns[1])
            players[turns[0]]=(player_opt,'optimal_player')
            players[turns[1]]=(agent_player,'agent')
        else:
            player_opt = OptimalPlayer(epsilon=eps_optimalplayer,player=turns[0])
            players[turns[0]]=(agent_player,'agent')
            players[turns[1]]=(player_opt,'optimal_player')
        
        for j in range(9):    
            
            #-- Get turn
            turn = env.current_player
            
            #-- observe grid
            grid,end,_ = env.observe() 
            
            #-- Play
            current_player, _ = players[turn]
            move = current_player.act(grid) 
            env.step(move,print_grid=False)
        
            #-- Chek that the game has finished
            if env.end :
                if env.winner is not None :
                    _,winner = players[env.winner]
                    wins_count[winner] = wins_count[winner] + 1
                else :
                    wins_count['draw'] = wins_count['draw'] + 1
                
                break
    
    M = (wins_count['agent']-wins_count['optimal_player'])/500
    
    if verbose :
        string ="M_rand"
        if eps_optimalplayer < 1:
            string = "M_opt"    
        print(string+" : ",M)
        print(wins_count)

    
    return M


def q_learning(epsilon,num_episodes:int,env:TictactoeEnv,path_save:str,eps_opt=0.5,alpha=0.05,gamma=0.99,render=False,test=False):
    
    q_table = dict()
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
    
    for episode in range(1,num_episodes+1):
        
        if episode % 250 == 0 :
            if episode % 1000 == 0 :
                print(f"\nEpisode : {episode}")
                print(wins_count)
            agent_mean_rewards[int(episode//250)-1] = accumulate_reward/250
            accumulate_reward = 0 # reset
            
            if test:
                M_opt = test_policy(0,q_table,verbose=True)
                M_rand = test_policy(1,q_table,verbose=True)
                M_opts.append(M_opt)
                M_rands.append(M_rand)
                
        env.reset()
        turns = turns[np.random.permutation(2)]
        player_opt = OptimalPlayer(epsilon=eps_opt,player=turns[0])
        agent_learner = turns[1]
        
        players[turns[0]]='optimal_player'
        players[turns[1]]='agent'
        
        current_state = None
        next_state = None
        A = None
        
        for j in range(9) : # The game takes at most 9 steps to finish

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
                #availables_positions = get_available_positions(current_state)
                #rnd = np.random.uniform(0,1)
                #if rnd > epsilon(episode) :
                #    A = get_best_action(availables_positions,q_table[current_state_hash]) # best move among possible moves        
                #else:
                #    np.random.shuffle(availables_positions) # shuffle
                #    A = availables_positions[0] # take first
                #    A = A.item()
                A = get_action(epsilon(episode), current_state, current_state_hash,q_table)
                
                #----- Take action A & Observe reward
                next_state,end,_ = env.step(A,print_grid=False)
                reward = env.reward(agent_learner)
                if reward > 0:  
                    #----- Update q_table when agent wins
                    q_table = update_player(q_table,current_state,env.observe()[0],A,alpha,reward,gamma)
                    
                #----- Update q_table
                #q_table = update_player(q_table,current_state,next_state,A,alpha,reward,gamma)

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
                    
                #-- Updating Q-value when agent looses
                #if env.reward(agent_learner) == -1:  
                    #----- Update q_table
                    #q_table = update_player(q_table,current_state,env.observe()[0],A,alpha,-1,gamma)
                    
                #-- accumulate rewards
                accumulate_reward += env.reward(agent_learner)
                    
                break
            
        #print(episode,env.winner)
        
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
    q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_1,num_episodes=int(20e3),env=env,path_save=path,alpha=0.05,gamma=0.99,render=False,test=False)
    plt.plot(agent_mean_rewards)
    plt.title(f'Average rewards with Epsilon {eps_1(0)}')

#-- Q.2 and Q.3
eps_min=0.1
eps_max=0.8

if False :
    fig,axs = plt.subplots(1,2)
    env = TictactoeEnv()
    test=True
    for N_star in [1,10e3,20e3,30e3,40e3]:
        eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
        q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_2,num_episodes=int(20e3),env=env,path_save='./20k_decrease.pickle',alpha=0.05,gamma=0.99,render=False,test=test)
        axs[0].plot(agent_mean_rewards,label=f'N* {N_star}')
        if test :
            axs[1].plot(M_opts,label=f' M_opt N* {N_star}')
            axs[1].plot(M_rands,label=f' M_rand N* {N_star}')

    #print('\nfinal : ',wins_count)
    axs[0].set_title('Average rewards')
    axs[1].set_title(r'$M_{opt}$ and $M_{rand}$')
    plt.legend()
    
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
    for eps in [0,0.25,0.5,0.75,1]:
        eps_2=lambda x : max([eps_min,eps_max*(1-x/N_star)])
        q_table,wins_count,agent_mean_rewards,M_opts,M_rands = q_learning(epsilon=eps_2,eps_opt=eps,num_episodes=int(20e3),env=env,path_save='./20k_Q4.pickle',alpha=0.05,gamma=0.99,render=False,test=test)
        plt.plot(M_opts,label=f'M_opt; eps_opt={eps}')
        plt.plot(M_rands,label=f'M_rand; eps_opt={eps}')

#-- Test
if True :
    env = TictactoeEnv()
    q_table = None
    with open(path,'rb') as file:
        q_table = pickle.load(file)
    
    player = agent(q_table)
    for eps in [0,1]:
        test_policy(eps,q_table,verbose=True)








