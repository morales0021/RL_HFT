from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import pdb
import pytz
import ray
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms import PPOConfig
from ray.tune.registry import register_env
from hftrl.env.callbacks.simplecallback import MyCustomCallbacks
from hftrl.env.config.simple_configs import ConfigMarketMakingEnv
from hftrl.env.physics.marketmaking_mt5 import MarketMakingEnv
from futsimulator.data_readers.mt5_redis import MT5RedisReader

import logging
import os

logger = logging.getLogger(__name__)

project_path = '/home/mora/Documents/projects/HFTRL/hftrl/tests/market_making_mt5/'
unit_space = Box(low = -np.inf,
                 high = np.inf,
                 shape = (1,),
                 dtype = np.float32)

host_redis = '192.168.1.48'
port_redis = 6379

ray.init(local_mode=True)

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
        "year": 2023,
        "month_day":{7:[17,5,28,10,25,3,6,13,19,20,18,12,26,4]},
        "preload_hour":14,
        "start_hour" : 14,
        "preload_minute": 1,
        "end_hour" : 16,
        "minutes_to_add" : 60
    },
    rwd_manager={
        "weights":{
            "instant_clpnl": 1,
            "o_pnl": 0.2
        }
    },
    tracker={},
    commission_cfg={},
    redis_host = '192.168.1.48',
    redis_port = 6379,
    decimal_time = 1e3,
    ticker = 'EP',
    suffix_ticker = 'idx',
    max_open_pos = 6,
    unit_tick = 0.25,
    enable_render=False,
    path_render = project_path,
    mt5_reader = None
)


cfg_mm_mt5_eval = ConfigMarketMakingEnv(
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
        "year": 2023,
        "month_day":{7:[21,14,7,27,11]},
        "preload_hour":14,
        "start_hour" : 14,
        "preload_minute": 1,
        "end_hour" : 16,
        "minutes_to_add" : 60
    },
    rwd_manager={
        "weights":{
            "instant_clpnl": 1,
            "o_pnl": 0.2
        }
    },
    tracker={},
    commission_cfg={},
    redis_host = '192.168.1.48',
    redis_port = 6379,
    decimal_time = 1e3,
    ticker = 'EP',
    suffix_ticker = 'idx',
    max_open_pos = 6,
    unit_tick = 0.25,
    enable_render=False,
    path_render = project_path,
    mt5_reader = None
)

register_env("MarketMakingEnv", lambda config: MarketMakingEnv(config))

config = (
    PPOConfig().
    resources(
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
        num_gpus= 0
    ).environment(
        env = MarketMakingEnv,
        env_config = cfg_mm_mt5.model_dump()
    ).rollouts(
        num_envs_per_worker = 2,
        num_rollout_workers = 20,
        batch_mode = "truncate_episodes"
    ).training(
        lr=5e-5,
        train_batch_size = 120*10,
        sgd_minibatch_size = 256,
        num_sgd_iter = 10,
        model = {"fcnet_hiddens": [256, 256]
                 },
        gamma = 0.99
    ).evaluation(
        evaluation_duration = 1,
        evaluation_duration_unit = "episodes",
        evaluation_interval = 2,
        evaluation_config={
            "callbacks": MyCustomCallbacks,
            "env_config": cfg_mm_mt5_eval.model_dump()
        }
    ).reporting(
        min_train_timesteps_per_iteration = 1000,
        min_sample_timesteps_per_iteration = 1000,
    ).framework('torch')
)

stop = {
        "training_iteration":1000000,
        "timesteps_total":100,
        "episode_reward_mean":100
        }

algo = config.build()

# Set mean
best_mean = -np.inf

for idx in range(stop['training_iteration']):
    result = algo.train()
    new_mean = result['env_runners']['episode_reward_mean']
    print("idx: ", idx, " mean: ", new_mean)
    
    # Saving the best trained model
    if new_mean and best_mean < new_mean:

        best_mean = new_mean
        print("Best mean ", best_mean)
        os.makedirs(os.path.join(project_path, "checkpoint"), exist_ok = True)
        algo.save(checkpoint_dir = os.path.join(project_path, "checkpoint"))
        file_name = os.path.join(project_path,"checkpoint","scores.txt")

        with open(file_name, 'a') as file:
            file.write(str(best_mean)+'\n')

algo.stop()

print('Finished training')