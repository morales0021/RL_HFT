import random
import numpy as np
from hftrl.env.physics.base import TradingEng
from futsimulator.manager.manager import PositionManager
from futsimulator.positions.position import SideOrder
from futsimulator.utils.performance import get_total_pnl
from futsimulator.utils.plotting import get_plot
from hftrl.env.utils.dategeneration import get_random_date
from futsimulator.data_readers.mt5_redis import MT5RedisReader
from hftrl.env.observations.obs_manager import ObsManager
from futsimulator.market.mt5snapshots import MT5Snapshot
from hftrl.env.tracker.tracker import Tracker
from hftrl.env.rewards.reward_manager import RewardManager
import talib
from talib import stream
from collections import deque
import os
import pdb

class TpSlDirectionalEnv(TradingEng):

    def __init__(self, config):

        self.config = config
        self.action_space = config["action_space"]
        self.observation_space = config["observation_space"]
        self.trading_days = config["trading_days"]
        self.comission_cfg = config["commission_cfg"]
        self.redis_host = config["redis_host"]
        self.redis_port = config["redis_port"]
        self.decimal_time = config["decimal_time"]
        self.ticker = config["ticker"]
        self.suffix_ticker = config["suffix_ticker"]
        self.unit_tick = config["unit_tick"]

        self.max_open_pos = config["max_open_pos"]

        self.enable_render= config['enable_render']
        self.path_render = config["path_render"]
        
        # position manager instance
        self._ps = None
        # snapshot instance
        self._snapshot = None
        # index date instance
        self.mt5_reader = MT5RedisReader(
            host_redis=self.redis_host,
            port_redis=self.redis_port,
            identifier='EP'
        )
        
        # Render information
        self.profit_clpnl = []
        self.profit_opnl = []
        self.prev_clpnl = None
        self.cycles = 0
        self.tracker = Tracker(self.config['tracker'])
        self.rwd_manager = RewardManager(self.config['rwd_manager'])
        self.obs_manager = ObsManager(self.observation_space)
        self.start_time = None
        self.end_time = None
        self.is_eval = self.config['is_eval']

    def _preprocess_action(self, action):
        """
        Preprocess the action to be used in the step method
        """
        action = {
            "tp": action[0],
            "sl": action[1],
            "action_type": action[2],
        }
        return action

    def step(self, action):

        action = self._preprocess_action(action)
        self._process_action(action)
        self.tracker.update_hist_clpnl(self._ps.get_infos())
        # if self.cycles > 1000:
        #     print(f"Cycles reached 1000 for {self.start_time} and {self.end_time}") 

        reward = self._process_reward()
        new_obs = self._process_obs()
        is_done, info_done = self._process_dones()
        info = {**info_done}

        if self.enable_render:
            self.render_custom()

        if self.cycles >  60:
            is_truncated = True
        else:
            is_truncated = False

        self.total_reward += reward
        if is_done or is_truncated:
            if self.is_eval:
                print("Evaluation")
            if not self.is_eval:
                print("Training")
            print(f"Done/truncated for {self.start_time} and {self.end_time}, with cycles {self.cycles}")
            print(f"Total Reward: {self.total_reward}")

        # if self.total_reward > 100:
        #     pdb.set_trace()
            
        return new_obs, reward, is_done, is_truncated, info
    
    def _process_action(self, action):
        """
        actions 
            - TP Range
            - SL Range
            - Do nothing, Liquidate, Buy, Sell
        """
        # Do nothing, keep previous position running
        # if action['action_type'] == 0:
        #     self.update_until_new_cycle()
        #     return
        # Liquidate the current position
        # if action['action_type'] == 0:
        #     self._ps.liquidate()
        #     self._ps.cancel_all()
        #     self.update_until_new_cycle()
        #     return
        # Send market order, but liquidate all opened orders first
        if action['action_type'] == 1 or action['action_type'] == 2:
            self._ps.liquidate()
            self._ps.cancel_all()
            # buy
            if action['action_type'] == 1:
                tp = self._snapshot.ask + action['tp']*self.unit_tick 
                sl = self._snapshot.ask - action['sl']*self.unit_tick
                side = SideOrder.buy
            # sell
            elif action['action_type'] == 2:
                tp = self._snapshot.bid - action['tp']*self.unit_tick
                sl = self._snapshot.bid + action['sl']*self.unit_tick
                side = SideOrder.sell
            else:
                raise Exception("Wrong action type")
            self._ps.send_market_order(
                side = side,
                tp = tp,
                sl = sl,
                size = 1
            )
            self.update_until_new_cycle()
            return

    def update_until_new_cycle(self, cycle_secs = 60):

        for k in range(cycle_secs):
            self.update_until_new_second()
        self.cycles += 1

    def update_until_new_second(self):
        """
        Update the snapshot and the position manager until a new second
        It also appends historical data that will be used for the indicators
        """
        tot_vol_buy = 0
        tot_vol_sell = 0

        high = 0
        low = 999999999
        open = self._snapshot.price

        old_second = self._snapshot.datetime.second

        while self._snapshot.datetime.second == old_second:
            self._ps.step()
            if self._snapshot.side == 's':
                tot_vol_sell += self._snapshot.size
            elif self._snapshot.side == 'b':
                tot_vol_buy += self._snapshot.size
            last_price = self._snapshot.price
 
            if last_price > high:
                high = last_price
            if last_price < low:
                low = last_price

        # Append the aggregated volume of the last second
        self.tracker.update_hist_vol_sell(tot_vol_sell)
        self.tracker.update_hist_vol_buy(tot_vol_buy)
        self.tracker.update_hist_last_price(last_price)
        self.tracker.update_hist_open_price(open)
        self.tracker.update_hist_high_price(high)
        self.tracker.update_hist_low_price(low)

    def _process_obs(self):

        obs = self.obs_manager.get_obs(self._ps, self.tracker)

        return obs

    def _process_reward(self):

        if self._snapshot.rl.finished:
            self._ps.liquidate()
        elif self.cycles > 60:
            self._ps.liquidate()

        rwd = self.rwd_manager.get_reward(self._ps, self.tracker, self.cumulated_clpnl, self.cumulated_opnl)

        return rwd
        
    def _process_dones(self):
        
        trading_infos = self._ps.get_infos()
        clpnl = trading_infos['cl_pnl']
        infos = {}
        done = False

        if self._snapshot.rl.finished:
            done = True
            infos['info_done'] = {
                "cl_pnl": clpnl
            }
        
        return done, infos

    def reset(self, *, seed = None, options = None):
        """
        Creates a new position manager that handles all the 
        operations
        """
        _, self.start_time, self.end_time = get_random_date(**self.trading_days)
        # Snapshot instance
        self._snapshot = MT5Snapshot(
            host = self.redis_host,
            port = self.redis_port,
            symbol = self.ticker,
            mt5_reader=self.mt5_reader,
            start_time = self.start_time,
            end_time = self.end_time,
            )
        # Position manager instance
        self._ps = PositionManager(
            self._snapshot, self.max_open_pos, self.comission_cfg,
            indicators = {}
            )
        
        # Historical datafor indicators
        self.tracker.reset()

        self.update_until_new_cycle()
        
        observation = self._process_obs()
        infos = {}

        # Reset render
        self.profit_clpnl = []
        self.profit_opnl = []
        self.prev_clpnl = None
        self.cycles = 0
        self.total_reward = 0
        self.cumulated_clpnl = []
        self.cumulated_opnl = []
        return observation, infos

    def render_custom(self, iter_val: int = None):
        """
        This is a custom render method.
        """
        infos = self._ps.get_infos()
        clpnl = infos['cl_pnl']
        oppnl = infos['open_orders']['o_pnl']

        # clpnl = get_total_pnl(infos, closed = True)
        # oppnl = get_total_pnl(infos, closed = False)
        # if abs(oppnl) > 50:
        #     pdb.set_trace()
        self.profit_clpnl.append(clpnl)
        self.profit_opnl.append(oppnl)
        is_done, _ = self._process_dones()

        if is_done and iter_val:
            # Save the profit and loss
            
            infos = {
                "clpnl":self.profit_clpnl,
                "opnl": self.profit_opnl
            }

            directory = os.path.join(self.path_render,'evaluations',str(iter_val))
            os.makedirs(directory, exist_ok=True)
            pathfile = os.path.join(directory,f"iter_{iter_val}.png")
            get_plot(infos, "index", "profit", "performance", pathfile)
