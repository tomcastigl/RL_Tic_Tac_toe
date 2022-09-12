## Tabular Q-learning and DQN to master Tic-Tac-Toe
This project aims to train artificial agents that can play the famous game of Tic Tac Toe. The whole training process can be found in the "training_full_code.ipynb"
notebook, where we explored different approaches such as learning from experts and self-playing learning for both Q-learning and DQN. 

We measured the performances of our agents using 2 measures: the fraction of games won against the random player (1->optimal, 0->bad) and the fraction of games won
against an optimal player, since the optimal policy for TTic-Tac-Toe can be explicity implemented (0->optimal, -1->bad). below are shown the performances during training
of both Q-learning and DQN:
<p align="center">
  <img 
    width="600"
    height="400"
    src="https://github.com/tomcastigl/RL_Tic_Tac_toe/blob/master/imgs/q13.png"
  >
</p>
![image](https://github.com/tomcastigl/RL_Tic_Tac_toe/blob/master/imgs/q13.png)
![image](https://github.com/tomcastigl/RL_Tic_Tac_toe/blob/master/imgs/q17.png)

The best way to test the agent is to play against it by running the "interactive_play.py" script!
