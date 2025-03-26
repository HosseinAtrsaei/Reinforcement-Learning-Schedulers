import numpy as np
from environment.env import Environnement
from matplotlib import pyplot as plt
class ValueIterationAgent():

    def __init__(self, environment: Environnement, gamma: float = 0.99, convergence_threshold: float = 1e-5):
        """
        Class for Value Iteration Algorithm

        Parameters
        ----------
            - environment : Model for the State : (User 1: (buffer_1, battery_1, SNR_1), User 2 : (buffer_2, battery_2, SNR_2))
            - gamma : Discount Factor (default = 0.99)
            - convergence_threshold : Convergence Criterion (default = 1e-5)
        """

        self.env = environment
        self.gamma = gamma
        self.convergence_threshold = convergence_threshold

    def train(self,):
        """
        Train the Value Iteration Agent.
        """
        print("Value Iteration Training : Process Initiated ... ")
        V, policy, deltas_array = self.value_iteration()
        print("Value Iteration Training : Process Completed !")
        print(f'Value function: {V}')
        print(f'Policy: {policy}')

        plt.plot(deltas_array)
        plt.title("Convergence of Value Iteration Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
        plt.savefig("./figures/convergence_value_iteration.png")

        return V, policy

    def value_iteration(self,):
        """
        Value Iteration Algorithm.

        Returns:
            A tuple (V, policy) of the optimal value function and the optimal policy.
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
                A vector of length n_actions containing the expected value of each action from the given state.
            """
            # initialize the vector of action values
            A = np.zeros(n_actions)
            # loop over the actions we can take from the given state (state_index)
            for action_index in range(n_actions):
                # Get the list of possible next states with their corresponding transition probability and rewards in the form of a tuples (transition probability, next state, reward)

                # Compute the action values for action_index

            return A

        #Initialization value function
        V = np.ones(n_states)
        policy = np.zeros([n_states, n_actions])

        #While not optimal value function/optimal policy
        while True:
            # Stopping condition
            delta = 0

            # Update each state
            for state_index in range(n_states):
                # Do a one-step lookahead to find the best action

                # choose the action that maximizes the state-value function

                # Calculate delta across all states

                # Update the value function: Bellman optimality eqn

            # Check for convergence
            if delta < self.convergence_threshold:
                break

        # Create a deterministic policy using the optimal value function
        for state_index in range(n_states):
            # One step lookahead to find the best action for this state

            # Always take the best action

        return V, policy, delta_array
