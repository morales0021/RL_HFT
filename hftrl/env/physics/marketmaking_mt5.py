import random
import numpy as np
from hftrl.env.physics.base import TradingEng
from futsimulator.market.redissnapshots import TBBOSnapshot
from futsimulator.manager.manager import PositionManager
from futsimulator.interfaces.redisindex import IndexDateDay
from futsimulator.positions.position import SideOrder
from futsimulator.utils.performance import get_total_pnl
from futsimulator.indicators.profile import VolumeProfile
from futsimulator.indicators.traded_vol import TradedVolume
from futsimulator.indicators.price import CurrentPrice
from futsimulator.indicators.positions import CurrentPositions
from futsimulator.utils.plotting import get_plot
from hftrl.env.utils.dategeneration import get_random_date
import talib
from talib import stream
from collections import deque
import os
import pdb

class SimpleEnv(TradingEng):

    def __init__(self, config):

        self.config = config
        self.action_space = config["action_space"]
        self.observation_space = config["observation_space"]
        self.trading_days = config["trading_days"]
        self.comission_cfg = config["commission_cfg"]
        self.redis_host = config["redis_host"]
        self.redis_port = config["redis_port"]
        self.tick_decimal = config["tick_decimal"]
        self.ticker = config["ticker"]
        self.suffix_ticker = config["suffix_ticker"]
        self.unit_tick = config["unit_tick"]

        self.max_open_pos = config["max_open_pos"]

        self.enable_render= config['enable_render']
        self.path_render = config["path_render"]
        
        # Historical datafor indicators        
        self._hist_last_price = deque(maxlen = 100)
        self._hist_open_price = deque(maxlen = 100)
        self._hist_high_price = deque(maxlen = 100)
        self._hist_low_price = deque(maxlen = 100)
        self._hist_vol_sell = deque(maxlen = 100)
        self._hist_vol_buy = deque(maxlen = 100)


        # position manager instance
        self._ps = None
        # snapshot instance
        self._snapshot = None
        # index date instance
        self.idx_date_day = IndexDateDay(
            prefix = self.ticker, suffix = self.suffix_ticker,
            host = self.redis_host, port = self.redis_port)
        

        # Render information
        self.profit_clpnl = []
        self.profit_opnl = []

        self.prev_clpnl = None

    def step(self, action):

        self._process_action(action)

        reward = self._process_reward()
        new_obs = self._process_obs()
        is_done = self._process_dones()
        info = {}

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
        if opened_pos == self.max_open_pos:
            if pos_type == SideOrder.buy:
                self._ps.send_limit_order(SideOrder.sell, sell_limit_order_price, 1)
            elif pos_type == SideOrder.sell:
                self._ps.send_limit_order(SideOrder.buy, buy_limit_order_price, 1)
            else:
                raise Exception("Wrong position type")

            # update until a new second
            self.update_until_new_second()
            return
        
        # Send both limit orders        
        self._ps.send_limit_order(SideOrder.sell, sell_limit_order_price, 1)
        self._ps.send_limit_order(SideOrder.buy, buy_limit_order_price, 1)

        # update until a new second
        self.update_until_new_second()

    def update_until_new_second(self):
        """
        Update the snapshot and the position manager until a new second
        """
        tot_vol_buy = 0
        tot_vol_sell = 0

        high = 0
        low = 999999999
        open = self._snapshot.last

        old_second = self._snapshot.datetime.second

        while self._snapshot.datetime.second == old_second:
            self._snapshot.update()
            self._ps.update()
            if self._snapshot.side == 's':
                tot_vol_sell += self._snapshot.size
            elif self._snapshot.side == 'b':
                tot_vol_buy += self._snapshot.size
            last_price = self._snapshot.last
 
            if last_price > high:
                high = last_price
            if last_price < low:
                low = last_price

        # Append the aggregated volume of the last second
        self._hist_vol_sell.append(tot_vol_sell)
        self._hist_vol_buy.append(tot_vol_buy)
        
        # Create historical pipes of last, high, low and open prices
        self._hist_last_price.append(last_price)
        self._hist_high_price.append(high)
        self._hist_low_price.append(low)
        self._hist_open_price.append(open)

        self._compute_indicators()

    def _process_obs(self):

        info_pos = self._ps.get_infos()

        size_pos = np.array(info_pos['open_orders']['total_size']),
        buy_pos = np.array(info_pos['open_orders']['side'] == SideOrder.buy)
        sell_pos = np.array(info_pos['open_orders']['side'] == SideOrder.sell)
        opnl = np.array(info_pos['open_orders']['o_pnl'])
        cpnl = np.array(info_pos['cl_pnl'])


        indicators = {
            "ma_price_5": np.array(self.ma_5),
            "ma_price_10":  np.array(self.ma_10),
            "ma_price_20":  np.array(self.ma_20),
            "ma_price_50":  np.array(self.ma_50),
            "natr_5": np.array(self.atr_5),
            "natr_10": np.array(self.atr_10),
            "natr_20": np.array(self.atr_20),
            "natr_50": np.array(self.atr_50),
            "size_pos": size_pos,
            "buy_pos": buy_pos,
            "sell_pos": sell_pos,
            "opnl": opnl,
            "cpnl": cpnl

            }
        
        return indicators

    def _process_reward(self):

        if self._snapshot.rl.finished:
            self._ps.liquidate()
        
        infos = self._ps.get_infos()
        clpnl = get_total_pnl(infos, closed = True)
        oppnl = get_total_pnl(infos, closed = False)
        if not self.prev_clpnl:
            self.prev_clpnl = clpnl
        # Reward for profit
        # if clpnl>0:
        #     tot_pnl = 0.01
        # elif clpnl == 0:
        #     tot_pnl =0.0
        # else:
        #     tot_pnl = -0.01

        tot_pnl = clpnl - self.prev_clpnl
        # Penalisation for many opened positions
        #tot_op_pos = -1*len(infos['closed_orders'])*0.001
        #tot = tot_pnl + tot_op_pos
        self.prev_clpnl = clpnl

        return tot_pnl

    def _process_dones(self):

        if self._snapshot.rl.finished:
            return True
        else:
            return False

    def reset(self, *, seed = None, options = None):
        """
        Creates a new position manager that handles all the 
        operations
        """
        start_time_preload, start_time, end_time = get_random_date(**self.trading_days)

        profile = VolumeProfile(self.side_lad,self.side_lad,1/32)

        bid_sold = TradedVolume(
             size_up = self.side_lad, size_down=self.side_lad, tick_unit = 1/32,
             type = 'A', seconds = 10)
        
        ask_bought = TradedVolume(
             size_up = self.side_lad, size_down=self.side_lad, tick_unit = 1/32,
             type = 'B', seconds = 10)
        
        current_price = CurrentPrice(
            size_up = self.side_lad, size_down=self.side_lad, tick_unit = 1/32)

        current_pos = CurrentPositions(
            size_up = self.side_lad, size_down=self.side_lad, tick_unit = 1/32)

        indicators = {
            'bid_sold': bid_sold,
            'ask_bought': ask_bought,
            'profile': profile,
            'current_price': current_price
            }
        
        self._snapshot = TBBOSnapshot(
            self.redis_host, self.redis_port, decimal = self.tick_decimal,
            idx_date_day = self.idx_date_day, start_time = start_time, end_time = end_time,
            indicators = indicators, start_time_preload = start_time_preload
            )
        
        indicators_manager = {"current_pos":current_pos}
        
        self._ps = PositionManager(
            self._snapshot, self.max_size, self.comission_cfg,
            indicators = indicators_manager
            )

        observation = self._process_obs(self.obs_args)
        infos = {}
        self.reset_render()
        self.prev_clpnl = None

        return observation, infos

    def reset_render(self):

        self.profit_clpnl = []
        self.profit_opnl = []

    def render_custom(self, iter_val: int = None):
        """
        This is a custom render method.
        """
        infos = self._ps.get_infos()
        clpnl = get_total_pnl(infos, closed = True)
        oppnl = get_total_pnl(infos, closed = False)
        # if abs(oppnl) > 50:
        #     pdb.set_trace()
        self.profit_clpnl.append(clpnl)
        self.profit_opnl.append(oppnl)

        if self._process_dones() and iter_val:
            # Save the profit and loss
            
            infos = {
                "clpnl":self.profit_clpnl,
                "opnl": self.profit_opnl
            }

            directory = os.path.join(self.path_render,'evaluations',str(iter_val))
            os.makedirs(directory, exist_ok=True)
            pathfile = os.path.join(directory,f"iter_{iter_val}.png")
            get_plot(infos, "index", "profit", "performance", pathfile)


    def _compute_indicators(self):
        # Indicators
        self.ma_5 = stream.SMA(
            np.fromiter(self._hist_last_price),
            timeperiod=5
            )
        
        self.ma_10 = stream.SMA(
            np.fromiter(self._hist_last_price),
            timeperiod=10
            )
        
        self.ma_20 = stream.SMA(
            np.fromiter(self._hist_last_price),
            timeperiod=20
            )
        
        self.ma_50 = stream.SMA(
            np.fromiter(self._hist_last_price),
            timeperiod=50
            )

        self.atr_5 = stream.NATR(
            np.fromiter(self._hist_high_price),
            np.fromiter(self._hist_low_price),
            np.fromiter(self._hist_last_price),
            timeperiod=5
            )

        self.atr_10 = stream.NATR(
            np.fromiter(self._hist_high_price),
            np.fromiter(self._hist_low_price),
            np.fromiter(self._hist_last_price),
            timeperiod=5
            )
        
        self.atr_20 = stream.NATR(
            np.fromiter(self._hist_high_price),
            np.fromiter(self._hist_low_price),
            np.fromiter(self._hist_last_price),
            timeperiod=5
            )
        
        self.atr_50 = stream.NATR(
            np.fromiter(self._hist_high_price),
            np.fromiter(self._hist_low_price),
            np.fromiter(self._hist_last_price),
            timeperiod=5
            )