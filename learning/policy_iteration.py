import numpy as np
from environment.env import Environnement
from matplotlib import pyplot as plt
class PolicyIterationAgent():
    
    def __init__(self, environment: Environnement, gamma: float = 0.99, convergence_threshold: float = 1e-5):
        """
        Class for Value Iteration Algorithm

        Parameters
        ----------
            - environment : Model for the State : (User 1: (buffer1, channel 1), User 2 : (buffer2, channel2))
            - gamma : Discount Factor (default = 0.99)
            - convergence_threshold : Convergence Criterion (default = 1e-5)
        """
        
        self.env = environment
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold

    def policy_evaluation(self, policy):
            """
            Evaluate a policy given an environment and a full description of the environment's dynamics.
                       
            Returns:
            ------------
                Vector of length n_states representing the value function and an array of delta between each iteration (used for plotting the convergence curve).
            """
            
            # array to save the delta between each iteration
            delta_array = []
            
            # state space size
            n_states = self.env.state_space.n_states
            
            # Start with a random (all 0) value function,
            V = np.zeros(n_states)
            while True:
                delta = 0
                # For each state, perform a "full backup"
                for state_index in range(n_states):
                    v = 0
                    # Look at the possible next actions
                    for action_index, action_prob in enumerate(policy[state_index]):
                        # For each action, look at the possible next states and compute the value function

                    # How much our value function changed (across any states)
                    # delta = ...

                    # V[state_index] = ...
                        
                # Stop evaluating once our value function change is below a threshold",
                if delta < self.convergence_threshold:
                    break
            return np.array(V), delta_array
        

    def policy_iteration(self):
        """
        Policy Improvement Algorithm.
        
        Returns:
        ------------
            A tuple (V, policy, delta_array) of the optimal value function and the optimal policy as well as the array of delta between each iteration.
        """
        # array to save the delta between each iteration
        delta_array = []
        
        # state and action space sizes
        n_states = self.env.state_space.n_states
        n_actions = self.env.action_space.n_actions
        
        def one_step_lookahead(state_index: int, V: np.array(float)):
            """
            Function to calculate the value for all actions A for a given state s (state_index).
            
            Args:
                state_index: The state to consider (int)
                V: The value to use as an estimator, Vector of length n_states
            
            Returns:
            ------------
                A vector of length n_actions containing the expected value of each action from the given state.
            """
            # initialize the vector of action values
            A = np.zeros(n_actions)
            # loop over the actions we can take from the given state (state_index)
            for action_index in range(n_actions):
                # Get the list of possible next states with their corresponding transition probability and rewards in the form of a tuples (transition probability, next state, reward)

                # Compute the action values for action_index

            return A

        # Start with a random policy
        policy = np.ones([n_states, n_actions]) / n_actions
        print(f'Starting Policy: {policy}')
        
        # While not optimal policy
        while True:
            # Evaluate the current policy
            
            # Will be set to false if we make any changes to the policy
            policy_stable = True

            # For each state, do the following
            for state_index in range(n_states):
                # Compute the best action we would take under the current policy
                
                # Do a one-step lookahead to find the best action 

                # Greedily update the policy
                
            # If the policy is stable, then this is the optimal one
            if policy_stable:
                return V, policy, delta_array
            
    def train(self,):
        """
        Train the Policy Iteration Agent.
        
        Returns:
        ------------
            A tuple (V, policy) of the optimal value function and the optimal policy.
        """
        print("Policy Iteration Training : Process Initiated ... ")
        V, policy, delta_array = self.policy_iteration()
        print("Policy Iteration Training : Process Completed !")
        print(f'Policy: {policy}')
        
        plt.plot(delta_array)
        plt.title("Convergence of Policy Iteration Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
        plt.savefig("./figures/convergence_policy_iteration.png")
        
        return V, policy