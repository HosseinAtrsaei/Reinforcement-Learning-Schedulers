import numpy as np
from tqdm import tqdm
from environment.env import Environnement
from matplotlib import pyplot as plt


class QLearningAgent():
    """Class for Q-Learning & Double Q-Learning Algorithm"""

    def __init__(self, environment: Environnement, gamma:float = 0.99, learning_rate: float = 1e-2, initial_q_value: float = -1):
        """
        Class for Q-Learning Algorithm

        """
        
        self.environment = environment
        self.n_states = environment.state_space.n_states
        self.n_actions = environment.action_space.n_actions
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.Q_matrix = initial_q_value * np.ones((self.n_states, self.n_actions))
        self.initial_value_Q_matrix = initial_q_value

        self.policy = np.zeros((self.n_states, self.n_actions))
        
        
    def update_Q_matrix(self, reward: float, current_state: int, next_state: int, action: int) -> None:
        """Updates the Q-matrix using the Bellman equation.

        Args:
            reward (float): _description_
            current_state (int): _description_
            next_state (int): _description_
            action (int): _description_
        Returns:
            None
        """
        # The old Q value
        Q_matrix = self.Q_matrix
        # The new Q value (learned value)
        Q_matrix[current_state, action] = Q_matrix[current_state, action] + self.learning_rate * (reward + self.gamma * np.max(Q_matrix[next_state, :]) - Q_matrix[current_state, action])
        # Update the Q-matrix of the current state and action pair
        self.Q_matrix = Q_matrix

    def choose_action(self, state_index, epsilon):
        """
        Choose an action following Epsilon Greedy Policy.
        """
        if np.random.random() < epsilon:
            return self.environment.action_space.sample()
        else:
            return np.argmax(self.Q_matrix[state_index, :])
            
    def train(self, n_episodes: int = 2000, n_time_steps: int = 5000, epsilon_decay: float = 0.999, epsilon_min: float = 0.01):
        """
        Train the Q-Learning agent.
        
        Parameters
        ----------
            - n_episodes : Number of Episodes
            - n_time_steps : Maximum number of time steps per episode
            - epsilon_decay : Decay Rate for the Exploration Rate
            - epsilon_min : Minimum Exploration Rate
        """

        print("[INFO] Q-Learning Training: Process Initiated ... ")
        print(f'[INFO] The state space is of size {self.n_states * self.n_actions}.')
        
        avg_rewards = []
        
        epsilon = 1

        for episode in tqdm(range(n_episodes)):
            
            # Generate a Random Initial State
            _ = self.environment.reset()

            percentage_unvisited_states = self.computes_percentage_unvisited_states()

            reward_episode = []
            
            for time_step in range(n_time_steps):
                # Get the State Index
                current_state = self.environment.state_space.get_state_index()
                # Choose an action following Epsilon Greedy Policy
                action = self.choose_action(current_state, epsilon)
                # Update State
                next_state, reward = self.environment.step(action)
                reward_episode.append(reward)
                # Update Q(s, a)
                self.update_Q_matrix(reward, current_state, next_state, action)
            avg_reward = np.mean(reward_episode)
            avg_rewards.append(avg_reward)
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            print(f'[INFO] Episode: {episode+1}/{n_episodes}, Average Reward: {avg_reward}, Epsilon: {epsilon:.2f}, Percentage of unvisited states: {percentage_unvisited_states:.2f}')
            print('---------------------------------------------------')

        print("[INFO] Q-Learning Training : Process Completed !")
        
        # Extract policy
        for s in range(self.n_states):
            best_action_index = np.argmax(self.Q_matrix[s, :])
            self.policy[s, best_action_index] = 1
        print(f"[INFO] Policy : {self.policy}")

        plt.plot(avg_rewards)
        plt.title("Convergence of Value Iteration Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.savefig("./figures/convergence_q_learning.png")
    
    def computes_percentage_unvisited_states(self) -> float:
        """Computes the percentage of unvisited states in the Q-matrix.

        Returns:
            percentage_unvisited_states (float): The percentage of unvisited states in the Q-matrix.
        """
        # compute the number of unvisited states
        number_of_unvisited_states = np.count_nonzero(self.Q_matrix == self.initial_value_Q_matrix)
        # compute the percentage of unvisited states
        percentage_unvisited_states = 100 * number_of_unvisited_states / self.Q_matrix.size
        return percentage_unvisited_states