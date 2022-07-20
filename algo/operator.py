"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("Operator", )

import util
from . import aux_functions, Agent
from collections import deque
import pickle
import numpy as np
import torch
import torch.optim as optim
import os
import random
import astral
from astral import sun


class Operator(util.OperatorBase):
    def __init__(self, energy_src_id, weather_src_id, buffer_len='48', p_1='1', p_0='1', history_modus='daylight', power_td=0.17, weather_dim=6, data_path="data"):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.observer = astral.Observer(latitude=51.34, longitude=12.38)

        self.energy_src_id = energy_src_id
        self.weather_src_id = weather_src_id

        self.weather_same_timestamp = []

        self.buffer_len = int(buffer_len)
        self.history_power_len = int(10000/float(power_td)) # power_td is the time difference between two consecutive power values
        self.replay_buffer = deque(maxlen=self.buffer_len)
        self.power_history = deque(maxlen=self.history_power_len) 
        self.daylight_power_history = deque(maxlen=int(self.history_power_len/2))
        self.history_modus = history_modus

        self.p_1 = int(p_1)
        self.p_0 = int(p_0)
        
        self.agents = deque(maxlen=4)
        self.policy = Agent.Policy(state_size=weather_dim) # If we keep track of time, temp, humidity, uv-index, precipitation and clouds we have weather_dim=6.
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.power_history_means = []
        self.power_lists = []
        self.actions = []
        self.rewards = []
        self.weather_data = []

        self.power_lists_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_power_lists_{self.p_1}_{self.p_0}_{self.history_modus}.pickle'
        self.actions_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_actions_{self.p_1}_{self.p_0}_{self.history_modus}.pickle'
        self.rewards_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_rewards_{self.p_1}_{self.p_0}_{self.history_modus}.pickle'
        self.weather_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_weather_{self.p_1}_{self.p_0}_{self.history_modus}.pickle'

        self.model_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_model_{self.p_1}_{self.p_0}_{self.history_modus}.pt'

        #if os.path.exists(self.model_file):
        #    self.policy.load_state_dict(torch.load(self.model_file))

    def run_new_weather(self, new_weather_data):
        new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)
        self.weather_data.append(new_weather_array)
        new_weather_input = np.mean(new_weather_array, axis=0)

        self.policy.eval()      # The current policy is used for prediction.
        with torch.no_grad():
            input = torch.from_numpy(new_weather_input).float().unsqueeze(0)
            output = torch.argmax(self.policy(input)).item()
        self.policy.train()

        if len(self.agents) < 4:
            self.agents.append(Agent.Agent())
        elif len(self.agents) == 4:
            oldest_agent = self.agents.popleft()
            self.agents.append(Agent.Agent())
            if len(self.replay_buffer)==self.buffer_len:
                random.shuffle(self.replay_buffer)
                for agent in self.replay_buffer:
                    agent.action, agent.log_prob = agent.act(self.policy)
                    if self.history_modus=='all':
                        agent.reward = agent.get_reward(agent.action, self.p_1, self.p_0, self.power_history)
                    elif self.history_modus=='daylight':
                        agent.reward = agent.get_reward(agent.action, self.p_1, self.p_0, self.daylight_power_history)
                    agent.learn(agent.reward, agent.log_prob, self.optimizer)

                oldest_agent.action, oldest_agent.log_prob = oldest_agent.act(self.policy)
                if self.history_modus=='all':
                    oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, self.p_1, self.p_0, self.power_history)
                elif self.history_modus=='daylight':
                    oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, self.p_1, self.p_0, self.daylight_power_history)
                oldest_agent.learn(oldest_agent.reward, oldest_agent.log_prob, self.optimizer)

                self.power_lists.append(oldest_agent.power_list)
                self.actions.append(oldest_agent.action)
                self.rewards.append(oldest_agent.reward)
            self.replay_buffer.append(oldest_agent)    
            
        torch.save(self.policy.state_dict(), self.model_file)
        
        with open(self.power_lists_file, 'wb') as f:
            pickle.dump(self.power_lists, f)
        with open(self.actions_file, 'wb') as f:
            pickle.dump(self.actions, f)
        with open(self.rewards_file, 'wb') as f:
            pickle.dump(self.rewards, f)
        with open(self.weather_file, 'wb') as f:
            pickle.dump(self.weather_data, f)

        newest_agent = self.agents[-1]
        newest_agent.save_weather_data(new_weather_input)
    
        return output

    def run_new_power(self, new_power_data):
        time, new_power_value = aux_functions.preprocess_power_data(new_power_data)

        self.power_history.append(new_power_value)

        for agent in self.agents:
            agent.update_power_list(new_power_value)

        sunrise = sun.sunrise(self.observer, date=time, tzinfo='Europe/Berlin')
        sunset = sun.sunrise(self.observer, date=time, tzinfo='Europe/Berlin') 
        if (sunrise<time) and (time<sunset):
            self.daylight_power_history.append(new_power_value)

    def run(self, data, selector):
        if os.getenv("DEBUG") is not None and os.getenv("DEBUG").lower() == "true":
            print(selector + ": " + str(data))
        if selector == 'weather_func':
            if self.weather_same_timestamp != []:
                if data['weather_time'] == self.weather_same_timestamp[-1]['weather_time']:
                    self.weather_same_timestamp.append(data)
                elif data['weather_time'] != self.weather_same_timestamp[-1]['weather_time']:
                    new_weather_data = self.weather_same_timestamp
                    output = self.run_new_weather(new_weather_data)
                    self.weather_same_timestamp = [data] 
                    return output
            elif self.weather_same_timestamp == []:
                self.weather_same_timestamp.append(data)
        elif selector == 'power_func':
            self.run_new_power(data)
