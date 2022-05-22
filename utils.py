import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random



def play_game(p1,p2):
    env=TictactoeEnv()
    grid, _, __ = env.observe()
    for j in range(9):
        if env.current_player == p1.player:
            move = p1.act(grid)
        else:
            move = p2.act(grid)

        grid, end, winner = env.step(move, print_grid=False)
        if end:
            return env
            break

def Metric(policy,optimal=False):
    N_wins=0
    N_losses=0
    N=0
    for _ in range(5):
        np.random.seed()
        for i in range(500):
            Turns = np.array([['X','O']]*250+[['O','X']]*250)
            if optimal:
                player_test = OptimalPlayer(epsilon=0., player=Turns[i,1])
            if not optimal:
                player_test = OptimalPlayer(epsilon=1., player=Turns[i,1])

            player_new = policy(player=Turns[i,0])
            env_end= play_game(player_new,player_test)
            if env_end.winner==player_new.player:
                N_wins+=1
            if env_end.winner==player_test.player:
                N_losses+=1
            N+=1

    return (N_wins - N_losses)/N


############################ Learning from experts
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
    
    if next_state is not None :
        #----- Update q_table
        array_of_q_values = q_table[current_state_hash]
        q_value = array_of_q_values[:,A]
        updated_q_value = q_value + alpha*(reward + gamma*get_max_Q(q_table,next_state) - q_value)
        array_of_q_values[:,A] = updated_q_value # update Q(S,A)
        q_table[current_state_hash] = array_of_q_values # store updated Q(S,A)
    
    else:
        #----- Update q_table
        array_of_q_values = q_table[current_state_hash]
        q_value = array_of_q_values[:,A]
        updated_q_value = q_value + alpha*(reward + gamma*0.0 - q_value)
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
        if grid_hash in self.q_table.keys():
            A = get_best_action(availables_positions,self.q_table[grid_hash])
        
        #-- For any other state not discovered, any available action can be taken
        else :
            np.random.shuffle(availables_positions)
            A = availables_positions[0].item()
            
        return A
    
####################################################################### 
class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()

        self.fc=nn.Sequential(nn.Linear(3*3*2,128),nn.ReLU(),
                              nn.Linear(128,128),nn.ReLU(),
                              nn.Linear(128,128),nn.ReLU(),
                              nn.Linear(128,9))
    
    def forward(self,x):
        x=x.view(-1,18)#x.sum(2).view(-1,9)
        x=self.fc(x)
        return x

class ReplayMemory(object):

    def __init__(self, capacity,batch_size):
        self.memory = deque([],maxlen=capacity)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.device='cpu'
        self.batch_size=batch_size
        self.step = 0
        self.accumulated_loss = 0
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

class deep_Q_player:
    def __init__(self):
        self.player=''
    
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

def test_policy(eps_optimalplayer,q_table=None,verbose=False,DQN_policy_net=None):
    
    env = TictactoeEnv() # environment
    
    if DQN_policy_net is None:
        assert q_table is not None, "Provide q_table"
        agent_player = agent(q_table) # agent    
        
    else:
        agent_player=deep_Q_player()
    
    #-- Holders 
    wins_count = dict()
    wins_count['optimal_player']=0
    wins_count['agent']=0
    wins_count['draw']=0 
    players = dict()
    players[None] = 'draw'
    turns = np.array(['X','O'])
    agent_symbol = None # 'X' or 'O'
    optimal_symbol = None
    num_illegal_steps = 0
    
    for episode in range(500):
        
        env.reset()
        np.random.seed(episode) 
        
        if episode < 250 :
            player_opt = OptimalPlayer(epsilon=eps_optimalplayer,player=turns[1])
            players[turns[0]]=(player_opt,'optimal_player')
            players[turns[1]]=(agent_player,'agent')
            agent_symbol = turns[1]
            optimal_symbol = turns[0]
            
        else:
            player_opt = OptimalPlayer(epsilon=eps_optimalplayer,player=turns[0])
            players[turns[0]]=(agent_player,'agent')
            players[turns[1]]=(player_opt,'optimal_player')
            agent_symbol = turns[0]
            optimal_symbol = turns[1]
            
        for j in range(9):    
            
            #-- Get turn
            turn = env.current_player
            
            #-- observe grid
            grid,end,_ = env.observe() 
            
            #-- Play
            current_player, _ = players[turn]
            
            #-- Playing with DQN-agent
            if q_table is None and turn==agent_symbol:
                state = grid2tensor(grid,agent_symbol)
                move = current_player.act(state,DQN_policy_net,0)
                
                try:
                    env.step(move,print_grid=False)
                    
                except ValueError:
                    env.end = True
                    env.winner = optimal_symbol
                    num_illegal_steps += 1

                
            else:
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
        print(wins_count,'\n')
        print('Number of illegal steps',num_illegal_steps)

    
    return M


