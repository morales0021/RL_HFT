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

class MarketMakingEnv(TradingEng):

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
        
        # Historical datafor indicators        
        # self._hist_last_price = deque(maxlen = 100)
        # self._hist_open_price = deque(maxlen = 100)
        # self._hist_high_price = deque(maxlen = 100)
        # self._hist_low_price = deque(maxlen = 100)
        # self._hist_vol_sell = deque(maxlen = 100)
        # self._hist_vol_buy = deque(maxlen = 100)

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

    def _preprocess_action(self, action):
        """
        Preprocess the action to be used in the step method
        """
        action = {
            "spread_limit_buy": action[0],
            "spread_limit_sell": action[1]
        }
        return action

    def step(self, action):

        action = self._preprocess_action(action)

        self._process_action(action)
        self.tracker.update_hist_clpnl(self._ps.get_infos())

        reward = self._process_reward()
        new_obs = self._process_obs()
        is_done, info_done = self._process_dones()
        info = {**info_done}

        if self.enable_render:
            self.render_custom()

        return new_obs, reward, is_done, is_done, info
    
    def _process_action(self, action):
        
        # Cancel all orders if there any limit orders opened
        self._ps.cancel_all()

        # Send new limit orders
        ask_price = self._snapshot.ask
        bid_price = self._snapshot.bid

        buy_limit_order_price = bid_price - action['spread_limit_buy']*self.unit_tick
        sell_limit_order_price = ask_price + action['spread_limit_sell']*self.unit_tick

        portfolio_state = self._ps.get_infos()
        opened_pos = portfolio_state['open_orders']['total_size']
        pos_type = portfolio_state['open_orders']['side']

        # If many opened position, only open one opposite limit order
        if opened_pos >= self.max_open_pos:
            if pos_type == SideOrder.buy:
                self._ps.send_limit_order(
                    price = sell_limit_order_price,
                    side = SideOrder.sell,
                    size = 1
                    )
            elif pos_type == SideOrder.sell:
                self._ps.send_limit_order(
                    price = buy_limit_order_price,
                    side = SideOrder.buy,
                    size = 1)
            else:
                raise Exception("Wrong position type")

            # update until a new second
            self.update_until_new_cycle()
            return
        
        # Send both limit orders        
        self._ps.send_limit_order(
            price = sell_limit_order_price,
            side = SideOrder.sell,
            size = 1
        )
        
        self._ps.send_limit_order(
            price = buy_limit_order_price, 
            side = SideOrder.buy,
            size = 1
        )
        # update until a new second
        self.update_until_new_cycle()

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

        # self._hist_vol_sell.append(tot_vol_sell)
        # self._hist_vol_buy.append(tot_vol_buy)
        
        # # Create historical pipes of last, high, low and open prices
        # self._hist_last_price.append(last_price)
        # self._hist_high_price.append(high)
        # self._hist_low_price.append(low)
        # self._hist_open_price.append(open)

        # self._compute_indicators()

    def _process_obs(self):

        obs = self.obs_manager.get_obs(self._ps, self.tracker)

        return obs
    
        # info_pos = self._ps.get_infos()
        # # pdb.set_trace()
        # size_pos = np.array([info_pos['open_orders']['total_size']])
        # buy_pos = np.array([info_pos['open_orders']['side'] == SideOrder.buy])
        # sell_pos = np.array([info_pos['open_orders']['side'] == SideOrder.sell])
        # opnl = np.array([info_pos['open_orders']['o_pnl']])
        # cpnl = np.array([info_pos['cl_pnl']])

        # indicators = {
        #     "ma_price_5": np.array([self.ma_5]),
        #     "ma_price_10":  np.array([self.ma_10]),
        #     "ma_price_20":  np.array([self.ma_20]),
        #     "ma_price_50":  np.array([self.ma_50]),
        #     "natr_5": np.array([self.atr_5]),
        #     "natr_10": np.array([self.atr_10]),
        #     "natr_20": np.array([self.atr_20]),
        #     "natr_50": np.array([self.atr_50]),
        #     "size_pos": size_pos,
        #     "buy_pos": buy_pos,
        #     "sell_pos": sell_pos,
        #     "opnl": opnl,
        #     "cpnl": cpnl
        #     }
        
        # return indicators

    def _process_reward(self):

        if self._snapshot.rl.finished:
            self._ps.liquidate()

        rwd = self.rwd_manager.get_reward(self._ps, self.tracker)

        return rwd
        
        # infos = self._ps.get_infos()
        # clpnl = infos['cl_pnl']
        # oppnl = infos['open_orders']['o_pnl']

        # if not self.prev_clpnl:
        #     self.prev_clpnl = clpnl

        # tot_pnl = clpnl - self.prev_clpnl

        # # Penalisation for many opened positions
        # self.prev_clpnl = clpnl

        # return tot_pnl + 0.2*oppnl

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
        _, start_time, end_time = get_random_date(**self.trading_days)
        # Snapshot instance
        self._snapshot = MT5Snapshot(
            host = self.redis_host,
            port = self.redis_port,
            symbol = self.ticker,
            mt5_reader=self.mt5_reader,
            start_time = start_time,
            end_time = end_time,
            )
        # Position manager instance
        self._ps = PositionManager(
            self._snapshot, self.max_open_pos, self.comission_cfg,
            indicators = {}
            )
        
        # Historical datafor indicators
        self.tracker.reset()
        # self._hist_last_price = deque(maxlen = 100)
        # self._hist_open_price = deque(maxlen = 100)
        # self._hist_high_price = deque(maxlen = 100)
        # self._hist_low_price = deque(maxlen = 100)
        # self._hist_vol_sell = deque(maxlen = 100)
        # self._hist_vol_buy = deque(maxlen = 100)

        self.update_until_new_cycle()
        
        observation = self._process_obs()
        infos = {}

        # Reset render
        self.profit_clpnl = []
        self.profit_opnl = []
        self.prev_clpnl = None
        self.cycles = 0
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


    # def _compute_indicators(self):

    #     # Indicators
    #     if len(self._hist_last_price) < 50:
    #         return

    #     self.ma_5 = stream.SMA(
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=5
    #         ) - self._hist_last_price[-1]
        
    #     self.ma_10 = stream.SMA(
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=10
    #         ) - self._hist_last_price[-1]
        
    #     self.ma_20 = stream.SMA(
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=20
    #         ) - self._hist_last_price[-1]
        
    #     self.ma_50 = stream.SMA(
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=50
    #         ) - self._hist_last_price[-1]
        
    #     self.atr_5 = stream.NATR(
    #         np.fromiter(self._hist_high_price, float),
    #         np.fromiter(self._hist_low_price, float),
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=5
    #         )

    #     self.atr_10 = stream.NATR(
    #         np.fromiter(self._hist_high_price, float),
    #         np.fromiter(self._hist_low_price, float),
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=5
    #         )
        
    #     self.atr_20 = stream.NATR(
    #         np.fromiter(self._hist_high_price, float),
    #         np.fromiter(self._hist_low_price, float),
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=5
    #         )
        
    #     self.atr_50 = stream.NATR(
    #         np.fromiter(self._hist_high_price, float),
    #         np.fromiter(self._hist_low_price, float),
    #         np.fromiter(self._hist_last_price, float),
    #         timeperiod=5
    #         )