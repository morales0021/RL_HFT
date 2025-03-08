from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import pdb
import pytz
import tqdm
from hftrl.env.config.simple_configs import ConfigMarketMakingEnv
from hftrl.env.physics.marketmaking_mt5 import MarketMakingEnv
from futsimulator.data_readers.mt5_redis import MT5RedisReader

unit_space = Box(low = -np.inf,
                 high = np.inf,
                 shape = (1,),
                 dtype = np.float32)

host_redis = '192.168.1.48'
port_redis = 6379

mt5_reader = MT5RedisReader(
    host_redis=host_redis,
    port_redis=port_redis,
    identifier='EP'
)

cfg_mm_mt5 = ConfigMarketMakingEnv(
    action_space = spaces.Dict({
        "spread_limit_buy":Discrete(20),
        "spread_limit_sell":Discrete(20),
        }),
    observation_space = spaces.Dict({
            "ma_price_5": unit_space, # moving average price - price
            "ma_price_10": unit_space,
            "ma_price_20": unit_space,
            "ma_price_50": unit_space,
            "natr_5": unit_space, # normalized average true range
            "natr_10": unit_space,
            "natr_20": unit_space,
            "natr_50": unit_space,
            "size_pos": unit_space,
            "buy_pos": unit_space,
            "sell_pos": unit_space,
            "opnl": unit_space,
            "cpnl": unit_space,
            }),
    trading_days={
        "year": 2024,
        "month_day":{11:[1]},
        "preload_hour":14,
        "start_hour" : 14,
        "preload_minute": 1,
        "end_hour" : 16,
        "minutes_to_add" : 1
    },
    commission_cfg={},
    redis_host = '192.168.1.48',
    redis_port = 6379,
    decimal_time = 1e3,
    ticker = 'EP',
    suffix_ticker = 'idx',
    max_open_pos = 6,
    unit_tick = 0.25,
    enable_render=False,
    path_render = '',
    mt5_reader = mt5_reader
)

mme = MarketMakingEnv(cfg_mm_mt5.model_dump())
obs, infos = mme.reset()

action = {
    "spread_limit_buy": 5,
    "spread_limit_sell": 3
}
for k in tqdm.tqdm(range(100)):
    obs, rwd, done, trunc, info = mme.step(action)
    print(obs)
    print(rwd)