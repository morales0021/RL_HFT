from collections import deque, defaultdict
import pdb
class Tracker:

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.hist_clpnl = deque(maxlen=1000)
        self.hist_last_price = deque(maxlen=1000)
        self.hist_open_price = deque(maxlen=1000)
        self.hist_high_price = deque(maxlen=1000)
        self.hist_low_price = deque(maxlen=1000)
        self.hist_vol_sell = deque(maxlen=1000)
        self.hist_vol_buy = deque(maxlen=1000)
        self.hist_vol = deque(maxlen=1000)

    def update_hist_clpnl(self, ps_infos):
        # pdb.set_trace()
        clpnl = ps_infos['cl_pnl']
        self.hist_clpnl.append(clpnl)

    def update_hist_last_price(self, last_price):
        self.hist_last_price.append(last_price)

    def update_hist_open_price(self, open_price):
        self.hist_open_price.append(open_price)

    def update_hist_high_price(self, high_price):
        self.hist_high_price.append(high_price)

    def update_hist_low_price(self, low_price):
        self.hist_low_price.append(low_price)

    def update_hist_vol_sell(self, vol_sell):
        self.hist_vol_sell.append(vol_sell)

    def update_hist_vol_buy(self, vol_buy):
        self.hist_vol_buy.append(vol_buy)
    
    def update_hist_vol(self, vol):
        self.hist_vol.append(vol)

    def get_hist_clpnl(self, idx):      
        if len(self.hist_clpnl) < abs(idx):
            return 0
        return self.hist_clpnl[idx]

    def get_hist_last_price(self):
        return self.hist_last_price
    
    def get_hist_open_price(self):
        return self.hist_open_price
    
    def get_hist_high_price(self):
        return self.hist_high_price
    
    def get_hist_low_price(self):
        return self.hist_low_price