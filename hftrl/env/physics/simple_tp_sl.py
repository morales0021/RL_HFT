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
import os
import pdb

class SimpleEnv(TradingEng):

    def __init__(self, config):

        self.config = config
        self.action_space = config["action_space"]
        self.observation_space = config["observation_space"]
        self.trading_days = config["trading_days"]
        self.tick_unit = config['tick_unit']
        self.comission_cfg = config["commission_cfg"]
        self.redis_host = config["redis_host"]
        self.redis_port = config["redis_port"]
        self.tick_decimal = config["tick_decimal"]
        self.ticker = config["ticker"]
        self.tp = config['tp']
        self.sl = config['sl']
        self.suffix_ticker = config["suffix_ticker"]

        self.max_size = config["max_size"]

        self.enable_render= config['enable_render']
        self.path_render = config["path_render"]
        

        # HARCODED TO BE MOVED TO CONFIG
        self.obs_args = {"norm_profile": 2000,
                         "bid_trans":200,
                         "ask_trans":200,
                         }
        self.side_lad = config['side_lad']

        # End of hardcoded

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
        new_obs = self._process_obs(self.obs_args)
        is_done = self._process_dones()
        info = {}

        if self.enable_render:
            self.render_custom()

        return new_obs, reward, is_done, is_done, info
    
    def _process_action(self, action):
        size = action["size"].item()

        if action["market_order"] == 0:
            tp = self._snapshot.bid - self.tp*self.tick_unit
            sl = self._snapshot.ask + self.sl*self.tick_unit
            self._ps.send_market_order(SideOrder.sell, size, tp = tp, sl = sl)
        elif action["market_order"] == 1:
            pass
        elif action["market_order"] == 2:
            tp = self._snapshot.ask + self.tp*self.tick_unit
            sl = self._snapshot.bid - self.sl*self.tick_unit
            self._ps.send_market_order(SideOrder.buy, size, tp = tp, sl = sl)

        # update the snapshot
        # update the position manager
        self._snapshot.update()
        self._ps.update()

    def _process_obs(self, norm_info):

        info_pos = self._ps.get_infos()

        # Volume profile
        profile = self._snapshot.indicators['profile'].profile
        val_dic = [val for key, val in profile.items()]
        val = np.array(val_dic)/norm_info['norm_profile']

        # Recent Buy/Sell
        bid_sold = self._snapshot.indicators['bid_sold'].volume
        val_dic = [val for key, val in bid_sold.items()]
        bid_trans = np.array(val_dic)/norm_info['bid_trans']

        ask_bought = self._snapshot.indicators['ask_bought'].volume
        val_dic = [val for key, val in ask_bought.items()]
        ask_trans = np.array(val_dic)/norm_info['ask_trans']
        
        current_price = self._snapshot.indicators['current_price'].price
        val_dic = [val for key, val in current_price.items()]
        price = np.array(val_dic)

        current_pos = self._ps.indicators['current_pos'].positions
        val_dic = [val for key, val in current_pos.items()]
        curr_pos = np.array(val_dic)

        closed_pos = [len(info_pos['closed_orders'])]
        closed_pos = np.array(closed_pos)

        # pdb.set_trace()
        # print(profile.keys())            
        # print(bid_sold.keys())            
        # print(ask_bought.keys())            
        # print(current_price.keys())
        # print(current_pos.keys())
        if val.shape[0] != 221:
            print(val.shape)
            raise Exception("Wrong shape val")
        if bid_trans.shape[0] != 221:
            print(bid_trans.shape)
            raise Exception("Wrong shape bid_trans")
        if ask_trans.shape[0] != 221:
            print(ask_trans.shape)
            raise Exception("Wrong shape ask_trans")
        if price.shape[0] != 221:
            print(current_price)
            print(price.shape)
            print(self._snapshot.datetime)
            raise Exception("Wrong shape price")
        if curr_pos.shape[0] != 221:
            print(curr_pos.shape)
            raise Exception("Wrong shape curr_pos")
        # print(bid_trans.shape)
        # print(ask_trans.shape)
        # print(price.shape)
        # print(curr_pos.shape)
        
        indicators = {
            "volume_profile": val,
            "bid_sold":  bid_trans,
            "ask_bought": ask_trans,
            "current_price": price,
            "current_pos": curr_pos,
            # "closed_pos": closed_pos
            }
        
        return indicators

    def _process_reward(self):

        if self._snapshot.rl.finished:
            self._ps.liquidate()
        
        infos = self._ps.get_infos()
        clpnl = get_total_pnl(infos, closed = True)

        # Reward for profit

        return clpnl

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