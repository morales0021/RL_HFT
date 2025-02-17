import random
from gymnasium import spaces
import gymnasium as gym
import random
import pdb
from typing import Any

class TradingEng(gym.Env):

    metadata = {"render_modes":["human"]}

    def __init__(self, config):
        
      self.config = config

    def reset(self):
      
      raise NotImplementedError()
    
    def step(self, action: Any):
      
      raise NotImplementedError()