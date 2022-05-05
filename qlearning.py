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
                #print(A)
        return A.item()
    

def get_max_Q(q_table:dict,state:np.array):
        
    hash_ = get_hash(state)
    max_q = 0.
    
    if hash_ in q_table.keys():
        max_q = np.max(q_table[hash_],axis=1)
        
    return max_q
    
#---------------------------------------------------
#################################################################################

def update_player(player,q_table,current_state,next_state,A,alpha,reward,gamma):
    
    current_state_hash = get_hash(current_state)
    
    #----- Update q_table
    array_of_q_values = q_table[current_state_hash]
    q_value = array_of_q_values[:,A]
    updated_q_value = q_value + alpha*(reward + gamma*get_max_Q(q_table,next_state) - q_value)
    array_of_q_values[:,A] = updated_q_value # update Q(S,A)
    q_table[current_state_hash] = array_of_q_values # store updated Q(S,A)
    
    return q_table

def q_learning(epsilon:float,num_episodes:int,env:TictactoeEnv,path_save:str,alpha=0.05,gamma=0.99,render=False):
    
    q_table = dict()
    turns = np.array(['X','O'])
    
    #-- Holder 
    wins_count = dict()
    wins_count['optimal_player']=0
    wins_count['agent']=0
    wins_count['draw']=0
    players = dict()
    players[None] = 'draw'
    
    accumulate_reward = 0
    agent_mean_rewards = [0]*int(num_episodes//250)
    
    for episode in range(1,num_episodes+1):
        
        if episode % 250 == 0 :
            #if render :
            print(f"\nEpisode : {episode}")
            print(wins_count)
            agent_mean_rewards[int(episode//250)-1] = accumulate_reward/250
            accumulate_reward = 0 # reset
            
        env.reset()
        turns = turns[np.random.permutation(2)]
        player_opt = OptimalPlayer(epsilon=0.5,player=turns[0])
        agent_learner = turns[1]
        
        players[turns[0]]='optimal_player'
        players[turns[1]]='agent'

        for j in range(9) : # The game takes at most 9 steps to finish

            #-- Optimal player plays 
            if env.current_player == turns[0]  :
                grid,end,_ = env.observe() #-- observe grid
                move = player_opt.act(grid) #-- get move
                env.step(move,print_grid=False) # optimal player takes a move
            
            #-- agent plays
            else:   
                
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
                        
                    break
                
                #-- Learning agent updates q_table
                current_state = env.observe()[0]
                current_state_hash = get_hash(current_state) 
                
                # Add current_state in q_table if applicable
                if not(current_state_hash in q_table.keys()): 
                    q_table[current_state_hash]=np.zeros((1,9))
                            
                #----- Choose action A with epsilon greedy
                #print(current_state)
                availables_positions = get_available_positions(current_state)
                rnd = np.random.uniform(0,1)
                
                if rnd > epsilon :
                    A = get_best_action(availables_positions,q_table[current_state_hash]) # best move among possible moves        
                else:
                    np.random.shuffle(availables_positions) # shuffle
                    A = availables_positions[0] # take first
                    A = A.item()
    
                #----- Take action A & Observe reward
                #print(f' episode {episode} ; Action :', A)
                #env.render()
                
    
                next_state,end,_ = env.step(A,print_grid=False)
                
                reward = env.reward(agent_learner)
                #----- Update q_table
                array_of_q_values = q_table[current_state_hash]
                q_value = array_of_q_values[:,A]
                updated_q_value = q_value + alpha*(reward + gamma*get_max_Q(q_table,next_state) - q_value)
                array_of_q_values[:,A] = updated_q_value # update Q(S,A)
                q_table[current_state_hash] = array_of_q_values # store updated Q(S,A)
            
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
                    
                break
            
        #print(episode,env.winner)
        
    #--save
    if path_save is not None:
        with open(path_save,'wb') as file:
            pickle.dump(q_table, file)
    
    return q_table,wins_count,agent_mean_rewards

env = TictactoeEnv()
q_table,wins_count,agent_mean_rewards = q_learning(epsilon=0.1,num_episodes=int(1e3),env=env,path_save='./20k_eps_01.pickle',alpha=0.05,gamma=0.99,render=False)

print('\nfinal : ',wins_count)
plt.plot(agent_mean_rewards)
plt.title('Average revards')