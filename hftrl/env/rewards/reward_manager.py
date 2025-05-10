import pdb
class RewardManager:

    def __init__(self, reward_config):
        self.reward_config = reward_config
        self.rwd_info = reward_config['weights']
        self.rwd_keys = reward_config['weights'].keys()

    def get_reward(self, ps_manager, tracker):

        tot_rwd = {}
        infos = ps_manager.get_infos()

        if 'instant_clpnl' in self.rwd_keys:
            i_clpnl = infos['cl_pnl'] - tracker.get_hist_clpnl(-2)
            tot_rwd['instant_clpnl'] = self.rwd_info['instant_clpnl'] * i_clpnl

        if 'o_pnl' in self.rwd_keys:
            tot_rwd['o_pnl'] = self.rwd_info['o_pnl'] * infos['open_orders']['o_pnl']
        # print(tot_rwd)
        # print(infos['cl_pnl'])
        return sum(tot_rwd.values())