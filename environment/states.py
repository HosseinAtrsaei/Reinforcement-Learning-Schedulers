import numpy as np
from .user import User
# from user import User

class State():
    """NOMA State Model (Buffer1, Buffer2, Channel1, Channel2, Battery1, Battery2)"""

    def __init__(self,  data_packets=0, 
                        maximum_number_of_packets=1,
                        snr_level=0, 
                        maximum_delay=1, 
                        maximum_battery_level=2, 
                        battery_level=0,
                        data_arrival_probability=0.5,
                        snr_levels_cardinality=3,
                        energy_arrival_probability=0.5,
                        n_users=2,
                        unavailable_action_penalty=5,
                ):
        
        # number of users
        self.n_users = n_users
        
        # Users initialization
        self.list_users = []
        for _ in range(n_users):
            user_k = User(data_packets=data_packets, 
                            maximum_number_of_packets=maximum_number_of_packets,
                            snr_level=snr_level, 
                            maximum_delay=maximum_delay, 
                            maximum_battery_level=maximum_battery_level, 
                            battery_level=battery_level,
                            data_arrival_probability=data_arrival_probability,
                            snr_levels_cardinality=snr_levels_cardinality,
                            energy_arrival_probability=energy_arrival_probability,
                            )  
            self.list_users.append(user_k)
            
        # state space cardinality
        self.n_states = self.nbr_possible_states()
        self.unavailable_action_penalty = unavailable_action_penalty
    
    def nbr_possible_states(self):
        """Get the number of possible states in the system"""
        n_states = 1
        for user_k in self.list_users:
            n_states *= user_k.n_states
            
        return n_states
    
    def initialize(self):
        """Initialize each user with a cleared buffer and packet arrival, as well as a channel gain"""

        for user_k in self.list_users:
            user_k.set_new_user_state()
    
    def update_state(self, is_action_possible):
        """Update the state of the system given an action and return the cost = number of packets delayed and discared"""
        state_rewards = []
        for user_k in self.list_users:
            user_reward = -1 * user_k.update_user_state()
            if not is_action_possible:
                # Add a penalty term in the reward for unavailable action
                user_reward = user_reward - self.unavailable_action_penalty
            state_rewards.append(user_reward * 2)
        
        return state_rewards

    def get_state(self):
        """Get the current state of the system"""

        state_dict = {}
        for k, user_k in enumerate(self.list_users):
            state_dict[f'user_{k+1}'] = user_k.get_user_state()
        
        return state_dict
    
    def get_state_index(self):
        """Get an index for the current state"""

        user_state_indices = []
        max_values = tuple()
        for user_k in self.list_users:
            user_state_indices.append(user_k.get_user_state_index())
            max_values = max_values + (user_k.n_states,)

        state_index = np.ravel_multi_index(user_state_indices, max_values)

        return state_index
    
    def get_state_from_index(self, state_index):
        """Set the current state using a state index"""
        users_state = []
        max_values = tuple()
        for user_k in self.list_users:
            max_values = max_values + (user_k.n_states,)

        user_state_indices = np.unravel_index(state_index, max_values)

        for user_k, user_state_index in zip(self.list_users, user_state_indices):
            users_state.append(user_k.get_user_state_from_index(user_state_index))

        return users_state
    
    def execute_action(self, list_actions, is_action_possible=True):
        if is_action_possible:
            for user_k, action_k in zip(self.list_users, list_actions):
                if action_k == 'idle':
                    continue
                else:
                    user_k.execute()
