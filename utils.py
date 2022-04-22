import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer

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
