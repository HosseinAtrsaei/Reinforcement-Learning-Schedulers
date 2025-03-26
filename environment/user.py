import numpy as np

class User():
    """This class represents a user in the NOMA system"""

    def __init__(self,  data_packets=0, 
                        maximum_number_of_packets=1,
                        snr_level=0, 
                        maximum_delay=1, 
                        maximum_battery_level=2, 
                        battery_level=0,
                        data_arrival_probability=0.5,
                        snr_levels_cardinality=3,
                        energy_arrival_probability=0.5,
                        maximum_energy_unit_arrival=1
                    ):
        
        # number of packets in the buffer
        self.data_packets = data_packets
        # Data arrival probability
        self.data_arrival_probability = data_arrival_probability
        # Maximum number of data packets arrived at once
        self.maximum_number_of_packets = maximum_number_of_packets
        # Maximum delay
        self.maximum_delay = maximum_delay      
        
        # Good, mid or bad channel
        self.snr_level = snr_level
        # Number of snr levels
        self.snr_levels_cardinality =  snr_levels_cardinality

        # Maximum energy units in the battery
        self.maximum_battery_level = maximum_battery_level
        # Battery level
        self.battery_level = battery_level
        # Energy arrival probability
        self.energy_arrival_probability = energy_arrival_probability
        # Maximum number of energy units arrived at once
        self.maximum_energy_unit_arrival = maximum_energy_unit_arrival

        # number of possible states
        self.n_states = (self.maximum_delay + 1) * self.snr_levels_cardinality * (self.maximum_battery_level + 1)
    
    def get_user_state_index(self):
        
        max_user_values = (self.maximum_delay + 1, self.snr_levels_cardinality, self.maximum_battery_level + 1)

        # data packet state index
        data_packets_index = self.data_packets
        # snr level state index
        snr_level_index = self.snr_level
        # battery level state index
        battery_level_index = self.battery_level

        user_state = [data_packets_index, snr_level_index, battery_level_index]

        # user state index
        user_state_index = np.ravel_multi_index(user_state, max_user_values)

        return user_state_index
    
    def get_user_state_from_index(self, user_state_index):

        max_user_values = (self.maximum_delay + 1, self.snr_levels_cardinality, self.maximum_battery_level + 1)
        
        [data_packets_index, snr_level_index, battery_level_index] = np.unravel_index(user_state_index, max_user_values)
        user_state = (data_packets_index, snr_level_index, battery_level_index)
        
        return user_state

    def set_new_user_state(self):
        """Initialize the user state by updating the number of packets in the buffer, the battery level and the channel SNR randomly"""

        # update the channel SNR
        self.snr_level = np.random.randint(self.snr_levels_cardinality)

        # update the number of packets in the buffer
        self.data_packets = np.random.binomial(self.maximum_number_of_packets, self.data_arrival_probability)

        # update the battery level
        self.battery_level = np.random.binomial(self.maximum_energy_unit_arrival, self.energy_arrival_probability)

    def update_user_state(self):
        """Update the user state by updating the number of packets in the buffer, the battery level and the channel SNR"""
        # so that we can use it to compute the cost/reward
        exceeded_delay = 0
        
        if self.data_packets == self.maximum_delay:
            exceeded_delay = 1

        # update the channel SNR
        self.snr_level = np.random.randint(self.snr_levels_cardinality)

        # update the number of packets in the buffer
        self.data_packets = self.data_packets + np.random.binomial(self.maximum_number_of_packets, self.data_arrival_probability)
        self.data_packets = min(self.data_packets, self.maximum_number_of_packets)

        # update the battery level
        self.battery_level = self.battery_level + np.random.binomial(self.maximum_energy_unit_arrival, self.energy_arrival_probability)
        self.battery_level = min(self.battery_level, self.maximum_battery_level)

        return exceeded_delay
    
    def execute(self):

        # as we can have at most one packet in the buffer, we execute it and consume energy
        if self.data_packets > 0 and self.battery_level > 0:
            # execute packet
            self.data_packets = self.data_packets - 1
            # consume energy
            self.battery_level = max(0, self.battery_level - 1)

    def __str__(self):
        """Print buffer information"""
        return f'Buffer state : {self.data_packets}, maximum battery level: {self.maximum_battery_level}, Number of Packets in the buffer : {self.data_packets}, Maximum Delay : {self.maximum_delay}, Current Battery level : {self.battery_level}, Maximum Battery level : {self.maximum_battery_level}, snr_level : {self.snr_level}'

    def get_user_state(self):
        return f'Buffer state : {self.data_packets}, SNR : {self.snr_level}, Current Battery level : {self.battery_level}'
