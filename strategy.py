import pandas as pd
from abc import ABC, abstractmethod

from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import TD3

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from gym.core import Env
from gym import spaces

import numpy as np

POSITION_LIMIT = 200
TRANSACTION_FEE = 1e-4
FEE_IN_BIPS = 4.0
PERCENTAGE_EXCHAGNE_VOLUME = 0.3
INITIAL_POSITION = 0
ROWS_NEEDED = 2*24*60

class StockEnv(Env):
    def __init__(self, training_data, start = None, end = None):
        super(StockEnv, self).__init__()
        
        self.training_data = training_data
        
        if start is None:
            self.start = training_data.index.min()
        else:
            self.start = start
        
        if end is None:
            self.end = training_data.index.max()
        else:
            self.end = end
        
        self.action_space = spaces.Box(-1, 1, shape=(1,))
        
        # maybe also add your action history?
        # position + ROWS_NEEDED x prices + ROWS_NEEDED x volume
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1 + ROWS_NEEDED, ))
        
        self.reward = 0
        self.turbulence = 0
        self.seed = 42
        self.fees = 0
        self.trades = 0
        self.cash = 0
        
        #todo: maybe shift it by ROWS_NEEDED to avoid zeros at the beggining
        self.train_idx = 0
        
        # todo: add more values to state
        self.state = [INITIAL_POSITION] + [0] * (ROWS_NEEDED)
        self.done = False
        self.ith_index_to_print = 10_000
        
        
    def step(self, action):
        #todo: turbulence
        current_observation = self.training_data.iloc[self.train_idx]
        if self.train_idx % self.ith_index_to_print == 0:
            print(self.train_idx)
        self.train_idx += 1
        current_date = current_observation.name
        
        current_volume = current_observation['volume']
        current_price = current_observation['price']
        
        if current_date >= self.end:
            self.done = True
            
        current_position = self.state[0]
        
        action *= POSITION_LIMIT
        
        target_position = min(POSITION_LIMIT, max(-POSITION_LIMIT, self.state[0] + action[0]))
        
        max_available_volume = PERCENTAGE_EXCHAGNE_VOLUME * current_volume
        
        volume_matched = min(max_available_volume, abs(target_position - current_position))
        
        direction = 1.0 if target_position > current_position else -1.0
        new_position = current_position + direction * volume_matched
        
        transaction_cost = direction * volume_matched * current_price
        fee = abs(transaction_cost) * TRANSACTION_FEE * FEE_IN_BIPS
        
        new_usd_cash = self.cash - transaction_cost - fee
        self.cash = new_usd_cash
        
        new_state = [new_position]
        new_state.extend(self.state[2:])
        new_state.append(current_price)
        
        self.state = new_state
        
        self.reward = new_usd_cash + new_position * current_price
        
        return self.state, self.reward, self.done, {}

    def reset(self):
        self.reward = 0
        self.turbulence = 0
        self.seed = 42
        self.fees = 0
        self.trades = 0
        self.cash = 0
        
        #todo: maybe shift it by ROWS_NEEDED to avoid zeros at the beggining
        self.train_idx = 0
        
        # todo: add more values to state
        self.state = [INITIAL_POSITION] + [0] * (ROWS_NEEDED)
        self.done = False
        return self.state
    
    def render(self, mode='human', close=False):
        return self.state
    


class Strategy(ABC):

    @abstractmethod
    def required_rows(self):
        raise NotImplementedError("Specify required_rows!")

    @abstractmethod
    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        assert len(current_data) == self.required_rows  # This much data will be fed to model

        return None  # If None is returned, no action is executed


class MeanReversionStrategy(Strategy):
    required_rows = 2*24*60   # minutes of data to be fed to model.

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        avg_price = current_data['price'].mean()
        current_price = current_data['price'][-1]

        target_position = current_position + (avg_price - current_price)/1000

        return target_position


# class YourStrategy(Strategy):
#     required_rows = 1
    
#     def __init__(self):
#         training_data = pd.read_pickle("data/train_data.pickle")
#         self.env = StockEnv(training_data)
        
#         self.model = A2C('MlpPolicy', self.env, verbose=0, device='cpu')
# #         self.model = PPO('MlpPolicy', self.env, verbose=0, device='cpu')
        
#         self.model.learn(total_timesteps = training_data.shape[0])
# #         self.model.learn(total_timesteps=100_000)
    
#     def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
#         state = [current_position]
#         state.extend(self.env.state[2:])
#         state.append(current_data['price'][0])
#         self.env.state = state
# #         state = [current_position] + current_data['price'].to_list()
        
#         action = self.model.predict(state)
        
#         target_position = current_position + action[0][0] * POSITION_LIMIT

#         return target_position


class YourStrategy(Strategy):
    required_rows = 34740  # minutes of data to be fed to model.

    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:

        avg_price = current_data['price'].mean()

        current_price = current_data['price'][-1]

        target_position = current_position + (avg_price - current_price)/1200

        return target_position