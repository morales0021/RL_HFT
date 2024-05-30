import random
from hftrl.env.physics.base import TradingEng
from futsimulator.market.redissnapshots import TBBOSnapshot
from futsimulator.manager.manager import PositionManager
from futsimulator.interfaces.redisindex import IndexDateDay

class Simple(TradingEng):

    def __init__(self, config):
        self.config = config
        self.action_space = config.action_space
        self.observation_space = config.observation_space
        self.trading_days = config.trading_days
        self.comission_cfg = config.comm_cfg
        self.redis_host = config.redis_conf
        self.redis_port = config.redis_port
        self.tick_decimal = config.tick_decimal
        self.ticker = config.ticker
        self.suffix_ticker = config.suffix_ticker

        self.max_b_size = config.max_b_size
        self.max_s_size = config.max_s_size

        # position manager instance
        self._ps = None
        # index date instance
        self.idx_date_day = IndexDateDay(
            prefix = self.ticker, suffix = self.suffix_ticker,
            host = self.redis_host, port = self.redis_port)

    def step(self, action):

        self._process_action(action)

        reward = self._process_reward(physic_state)
        new_obs = self._process_obs(physic_state)
        is_done = self._process_dones(physic_state)

        return new_obs, reward, is_done, is_done, info  
    
    def _process_action(self, action):
        pass


    def _process_obs(self, physic_state):
        pass

    def _process_reward(self, physic_state):
        pass

    def _process_dones(self, physic_state):
        pass

    def reset(self, *, seed = None, options = None):
        """
        Creates a new position manager that handles all the 
        operations
        
        """
        idx_days = len(self.trading_days)
        idx_day = random.choice(list(range(0,idx_days)))
        start_time, end_time = self.trading_days[idx_day]

        self.snapshot = TBBOSnapshot(
            self.redis_host, self.redis_port, decimal = self.tick_decimal,
            idx_date_day= self.idx_date_day, start_time = start_time, end_time = end_time)
        
        self._ps = PositionManager(self.snapshot, self.max_b_size, self.max_s_size, self.commission_cfg)