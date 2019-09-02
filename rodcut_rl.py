'''
    Traditional 'Rodcut' solved as a Reinforcement Learning problem.  

    Problem statement: 
        Given a rod of length 'n' and a list of prizes of rod of length 'i' (where 1 <= i <= n),
        find the optimal way to cut rod into smaller rods in order to maximize profit.
        
    RL Formulation:
        - Let the state signify the remaining legth of the rod. 
        - Action taken by the agent would be the "how" much should the cut's length be.
        - For example, we start with rod of length 'n'. So, the state in which we are initially in is 'n'. 
            Suppose a take an action to cut a piece of length 'k' where 0 <= k <= n ('0' signifying not to cut). 
            Ergo, the next state I'm in would be 'n-k'. 
        - The reward that I get for taking action 'k' from state 'n' is given by the 
            corresponding prize for rod of length 'k'. 
        - Since the state space satisfies the 'Markov Property' ("The future is independant of the past given the present"), 
            I can claim to have formulated a MDP. 

        Mathematically, 
            Env: The rod itself!
            State space (discrete): 1 ... n
            Action space (discrete): 1 ... n
            Reward space: set of Real Number (R) 
    
    Note that, here, we "know" the environment; by "knowing" I mean we know the matirx of state transition probabilities (one-hot)
    and I even know all the possible states that I can be in at every instant. So, I argue that the requirement of sampling 
    an episode and solving with a Monte_Carlo-esque approach is redundant. Instead a (model-based) Deterministic total_rewardue Iteration
    algorithm (by equipping the Bellman Optimality Equation) can be adopted to solve. There is no explicit policy that we're 
    improving, although, the (deterministic) policy can be finally attained by taking an argmax of the Q_table for all the states. 

'''

import numpy as np

class Environment:

    def __init__(self, n, bounty):
        self.state_space = range(0, n)
        self.action_space = range(0, n)
        self.bounty = bounty

    def step(self, current_state, action):
        assert action <= current_state

        reward = self.bounty[action]
        next_state = current_state - action

        return next_state, reward

if __name__ == "__main__":

    # inputs
    rod_length = 5
    prize = [1, 5, 4, 3, 2]
    debug = 1

    # init
    bounty = [0] + prize  # set reward for "zero" cut 
    env = Environment(n=rod_length+1, bounty=bounty)
    q_table = np.zeros((len(env.state_space), len(env.action_space)))
    discount_factor = 0.99
    n_iter = 100

    # Value Iteration
    while n_iter > 0:
        for state in env.state_space:
            for action in env.action_space: 
                if (action <= state): # cannot cut a length greater than the remaning length of the rod
                    next_state, reward = env.step(state, action)

                    # Bellman Optimality Equation
                    q_table[state, action] = reward + (discount_factor * np.max(q_table[next_state]))
                    
                    if (debug):
                        print("State: " + str(state) + ", Action: " + str(action))
                        print("--- Next state: " + str(next_state) + ", Reward: " + str(reward))
        
        n_iter -= 1

    if (debug):
        print("\nQ Table: \n", q_table)
    
    # Computing the optimal policy
    optim_policy = []
    for state in env.state_space:
        optim_policy.append(np.argmax(q_table[state]))

    print("\nOptimal Policy: ", optim_policy)

    # Computing the optimal cuts  
    optim_cuts = []
    curr_state = rod_length
    total_reward = 0
    while curr_state > 0:
        index = np.argmax(q_table[curr_state])
        total_reward += bounty[index]    
        curr_state = curr_state - index
        
        optim_cuts.append(index)
    
    print("\nOptimal cuts: ", optim_cuts)
    print("\nTotal reward attained: ", total_reward)
