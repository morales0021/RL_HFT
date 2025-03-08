from collections import deque, defaultdict

class Tracker:

    def __init__(self, config):
        self.config = config
        self.hist_clpnl = defaultdict(lambda: deque(maxlen=1000))

    def update_all(self, ps):
        ps_infos = ps.get_infos()
        self.update_hist_clpnl(ps_infos)

    def update_hist_clpnl(self, ps_infos):

        clpnl = ps_infos['cl_pnl']
        self.hist_clpnl['cl_pnl'].append(clpnl)


    def get_hist_clpnl(self, idx):
        
        if len(self.hist_clpnl) < abs(idx):
            return 0
        return self.hist_clpnl[idx]