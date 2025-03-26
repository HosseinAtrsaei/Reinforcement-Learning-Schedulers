import numpy as np
from .states import State
from .actions import Action
from .transitions import Transition
from .rewards import Reward
        

class Environnement():
    """Environnement class for the RL problem"""

    def __init__(self, states: State, actions: Action, transitions:Transition, rewards:Reward):

        self.state_space = states
        self.action_space = actions
        self.transition_model = transitions
        self.reward_model = rewards

    def p(self, state_index, action_index):
        """get the possible next states, their transition probabilities, and the reward"""
        transition = []
        
        reward = self.reward_model.reward_matrix[state_index, action_index]
        
        next_state_probs = self.transition_model.transition_matrix[state_index, action_index, :]
        
        # fill the transition list with the tuples (P[s'], s', r)
        for next_state_index, next_state_prob in enumerate(next_state_probs):
            # optional verification
            if next_state_prob == 0:
                continue
            transition.append((next_state_prob, next_state_index, reward))
        
        return transition
    
    def reset(self):
        self.state_space.initialize()
    
    def step(self, action_index):
        
        action_dictionary = self.action_space.get_action_dictionary(action_index)
        list_actions = list(action_dictionary.values())

        is_action_possible = self.is_action_possible(action_index)
        # Execute the action list for the state users
        self.state_space.execute_action(list_actions, is_action_possible)
        # Update the state information and transition to a new one
        state_rewards = self.state_space.update_state(is_action_possible)
        # compute the reward
        reward = sum(state_rewards)
        # transit to the next state
        next_state_index = self.state_space.get_state_index()

        return next_state_index, reward

    def is_action_possible(self, action_index):
        """Verify if the chosen action is possible to execute (taken from the `compute_reward` function, maybe should be put elsewhere)"""
        possible_action = True
        
        current_state_index = self.state_space.get_state_index()
        current_state_tuple = self.state_space.get_state_from_index(current_state_index)
        
        action_dictionary = self.action_space.get_action_dictionary(action_index)
        
        noma_users_snr = []
        
        # Return false if one of following conditions are met
        for l, user_current_state in enumerate(current_state_tuple):
            # the action number of packets to execute for the user are not available in the user buffer
            if action_dictionary[f'user_{l+1}'] > user_current_state[0]:
                possible_action = False
            
            if action_dictionary[f'action_{l+1}'] == 'communicate':
                # log the communicating users to verify multiple communications
                if user_current_state[1] > 1:
                    noma_users_snr.append(user_current_state[1])
                # the user action dictates that the user communicates, but its SNR does not allow it 
                if user_current_state[1] == 0:
                    possible_action = False
            
            # the user battery is not sufficient to execute the action number of packets
            if action_dictionary[f'user_{l+1}'] > user_current_state[2]:
                possible_action = False
        
        # the joint action dictates multiple communications
        if len(noma_users_snr) > 1:
            for user_snr in noma_users_snr:
                # one of the user SNRs does not permit it
                if user_snr < len(noma_users_snr):
                    possible_action = False
        
        return possible_action
