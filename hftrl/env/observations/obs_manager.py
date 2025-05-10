import numpy as np
from futsimulator.positions.position import SideOrder
from hftrl.env.observations.raw_obs import compute_delta_ma, compute_atr

class ObsManager:

    def __init__(self, obs_space):
        self.obs_space = obs_space
        self.obs_keys = obs_space.keys()

    def get_obs(self, ps_manager, tracker):

        obs = {}

        infos = ps_manager.get_infos()

        hist_last_price = tracker.get_hist_last_price()
        hist_open_price = tracker.get_hist_open_price()
        hist_high_price = tracker.get_hist_high_price()
        hist_low_price = tracker.get_hist_low_price()

        if 'ma_price_5' in self.obs_keys:
            obs['ma_price_5'] = compute_delta_ma(hist_last_price, 5)

        if 'ma_price_10' in self.obs_keys:
            obs['ma_price_10'] = compute_delta_ma(hist_last_price, 10)

        if 'ma_price_20' in self.obs_keys:
            obs['ma_price_20'] = compute_delta_ma(hist_last_price, 20)

        if 'ma_price_50' in self.obs_keys:
            obs['ma_price_50'] = compute_delta_ma(hist_last_price, 50)

        if 'natr_5' in self.obs_keys:
            obs['natr_5'] = compute_atr(
                hist_last_price, hist_high_price, hist_low_price, 5)

        if 'natr_10' in self.obs_keys:
            obs['natr_10'] = compute_atr(
                hist_last_price, hist_high_price, hist_low_price, 10)

        if 'natr_20' in self.obs_keys:
            obs['natr_20'] = compute_atr(
                hist_last_price, hist_high_price, hist_low_price, 20)

        if 'natr_50' in self.obs_keys:
            obs['natr_50'] = compute_atr(
                hist_last_price, hist_high_price, hist_low_price, 50)

        if 'size_pos' in self.obs_keys:
            obs['size_pos'] = infos['open_orders']['total_size']

        if 'buy_pos' in self.obs_keys:
            obs['buy_pos'] = infos['open_orders']['side'] == SideOrder.buy

        if 'sell_pos' in self.obs_keys:
            obs['sell_pos'] = infos['open_orders']['side'] == SideOrder.sell

        if 'opnl' in self.obs_keys:
            obs['opnl'] = infos['open_orders']['o_pnl']

        if 'cpnl' in self.obs_keys:
            obs['cpnl'] = infos['cl_pnl']

        obs = self.cast_obs(obs)
        #import pdb; pdb.set_trace()
        return obs
    
    def cast_obs(self, obs):
        for key in obs.keys():
            obs[key] = np.array([obs[key]], dtype=np.float32)
        return obs