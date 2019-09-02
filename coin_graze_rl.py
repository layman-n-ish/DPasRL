'''
    'Robot Coin Graze' problem solved as a Reinforcement Learning problem.

    Problem statement:
        Several coins are placed in cells of an n × m board, no more than one coin per cell. 
        A robot, located in the upper left cell of the board, needs to collect as many of the coins 
        as possible and bring them to the bottom right cell. On each step, the robot can move either
        one cell to the right or one cell down from its current location. When the robot visits a cell
        with a coin, it always picks up that coin. Design an algorithm to find the maximum number of coins 
        the robot can collect and a path it needs to follow to do this.

    RL Formulation:
        - Let the state signify my cell location (row_index, column_index) on the board.
        - It's a no-brainer that the allowed actions allowed from a particular state are 
            right (0) and down (1), as explicitly mentioned in the problem statement.
        - The reward attained by taking an action from a particular state is the amount
            of the coin present in the state reached. 
        - Since the state space satisfies the 'Markov Property' ("The future is independant of the past given the present"), 
            I can claim to have formulated a MDP.

        Mathematically, 
            Env: The board itself!
            State space (discrete): vector belonging to R^2
            Action space (discrete): 0 (right), 1 (down)
            Reward space: set of Real Number (R) 

    As seen in the other question, here, we "know" the environment; by "knowing" I mean we know the matirx of state 
    transition probabilities (one-hot) and I even know all the possible states that I can be in at every instant. 
    So, I argue that the requirement of sampling an episode and solving with a Monte_Carlo-esque approach is redundant. 
    Instead a (model-based) Deterministic total_rewardue Iteration algorithm (by equipping the Bellman Optimality Equation) 
    can be adopted to solve. There is no explicit policy that we're improving, although, the (deterministic) policy 
    can be finally attained by taking an argmax of the Q_table for all the states. 

    The state reached after taking an action which leads the agent to go outside the board will result in the next_state being 
    equal to the current_state. We can do this since we're ourselves modelling the environment to do so; which can be 
    mathematically potrayed in the matrix of state transition probabilites.

'''

import numpy as np

class Environment:

    def __init__(self, n, m):
        self.n_actions = 2
        self.n_states = n * m
        self.action_space = range(self.n_actions)
        self.state_space = np.array([(i, j) for i in range(n) for j in range(m)])

    def step(self, curr_state, action):

        # handling boundary conditions; giving negative reward to make sure it doesn't always get stuck there in the same state
        if curr_state[1] == m-1 and action == 0: 
            return curr_state, -5

        if curr_state[0] == n-1 and action == 1: 
            return curr_state, -5

        # normal conditions
        if action == 0:
            return [curr_state[0], curr_state[1]+1], coins_board[curr_state[0], curr_state[1]+1]

        if action == 1:
            return [curr_state[0]+1, curr_state[1]], coins_board[curr_state[0]+1, curr_state[1]]

def compute_optim_policy(q_table):

    policy = {}
    for i, j in env.state_space:
            max_q = -1000
            max_action_index = 0
            for a in env.action_space:
                if q_table[(i, j, a)] > max_q:
                    max_q = q_table[(i, j, a)]
                    max_action_index = a

            policy[(i, j)] = max_action_index
    
    return policy

if __name__ == "__main__":

    # init
    coins_board = np.array([[0, 3, 4, 5],
                            [2, 50, 4, 4],
                            [1, 40, 1, 3],
                            [5, 4, 3, 0]])

    n, m = coins_board.shape
    env = Environment(n, m)
    q_table = {(i, j, a): 0 for i in range(n) for j in range(m) for a in range(env.n_actions)} 
    discount_factor = 0.99
    n_iter = 1000
    debug = 1

    # Value Iteration
    while n_iter > 0:
        for state in env.state_space:
            for action in env.action_space:
                next_state, reward = env.step(state, action)

                # computing best q_val(curr_state) across all actions
                q_max_t = -1000
                for a in env.action_space: 
                    if q_table[(state[0], state[1], a)] > q_max_t:
                        q_max_t = q_table[(state[0], state[1], a)]
                
                # Bellman Optimality Equation
                q_table[(state[0], state[1], action)] = reward + (discount_factor * q_max_t)

                if(debug):
                    print("State: " + str(state) + ", Action: " + str(action))
                    print("--- Next state: " + str(next_state) + ", Reward: " + str(reward))

        n_iter -= 1

    # printing the converged Q_table
    if(debug):
        print("\nQ Table: ")
        for i, j in env.state_space:
            for a in env.action_space:
                print("(i, j, a): (", i, j, a, ") ; q_val = ", q_table[(i, j, a)])

    # computing the optimal policy by  acting greedy on the Q_table at every state 
    optim_policy = compute_optim_policy(q_table)
    print("\nOptimal policy: ")
    for i, j in env.state_space:
        print("(i, j): (", i, j, ") ; action = ", optim_policy[(i, j)])

    # Generating a simulation of the agent collecting coins on the board 
    print("\nRunning the policy...")
    curr_state = [0, 0]
    total_reward = 0
    for total_moves in range(n+m-2):

        # querying the best action based on the optimal policy
        best_action = optim_policy[(curr_state[0], curr_state[1])]

        # actually taking that step in the environment which sends back the observation and reward
        next_state, reward = env.step(curr_state, best_action)

        if best_action == 0:
            print ("State: ", curr_state, " -> Move right")
        else:
            print ("State: ", curr_state, " -> Move down")
        
        total_reward += reward
        curr_state = next_state

    print("Terminal state reached: ", curr_state)
    print("\nTotal reward gained: ", total_reward)
    print("\nClosing the environment...\n")
