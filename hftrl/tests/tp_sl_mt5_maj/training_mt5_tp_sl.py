from gymnasium import spaces
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import ray
from ray.rllib.algorithms import PPOConfig
from ray.tune.registry import register_env
from hftrl.env.config.simple_configs import ConfigMarketMakingEnv
from hftrl.env.physics.tp_sl_directional import TpSlDirectionalEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from hftrl.rllib.callbacks.callbacks import (
    SaveCheckpointData,
    log_multi_agent_episode_metrics,
    SaveBestMeanEpisode
)

from ray import tune, train
from hftrl.env.agents.mlp import MLPAgent

import logging
import os

logger = logging.getLogger(__name__)

project_path = '/home/mora/Documents/projects/HFTRL/hftrl/tests/tp_sl_mt5_maj/'
unit_space = Box(low = -np.inf,
                 high = np.inf,
                 shape = (1,),
                 dtype = np.float32)

host_redis = '192.168.1.48'
port_redis = 6379

ray.init(local_mode=False)

action_space_tp_sl = MultiDiscrete([20,20,2])

cfg_mm_mt5 = ConfigMarketMakingEnv(
    action_space = action_space_tp_sl,
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
        "month_day":{7:[3,4,5,6,7,10,11,12,13,14,17,18,19,20,21,24,25,26,27,28]},
        "preload_hour":14,
        "start_hour" : 14,
        "preload_minute": 1,
        "end_hour" : 21,
        "minutes_to_add" : 60
    },
    rwd_manager={
        "weights":{
            "instant_clpnl": 1,
            "o_pnl": 0.0
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
    mt5_reader = None,
    is_eval = False
)


cfg_mm_mt5_eval = ConfigMarketMakingEnv(
    action_space = action_space_tp_sl,
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
        "month_day":{8:[1,2,3,4,7,8,9,10,11,14,15,16,17,18,21,22,23,24,25,28,29]},
        # "month_day":{8:[9]},
        "preload_hour":14,
        "start_hour" : 14,
        "preload_minute": 1,
        "end_hour" : 21,
        "minutes_to_add" : 60
    },
    rwd_manager={
        "weights":{
            "instant_clpnl": 1,
            "o_pnl": 0.0
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
    mt5_reader = None,
    is_eval = True
)

register_env("TpSlDirectionalEnv", lambda config: TpSlDirectionalEnv(config))

config = (
    PPOConfig()
    .framework("torch")
    .environment(
        env = TpSlDirectionalEnv,
        env_config = cfg_mm_mt5.model_dump()
    )
    .env_runners(
        num_envs_per_env_runner = 1,
        num_env_runners = 25,
        sample_timeout_s = 30,
    )
    .training(
        lr=5e-5,
        train_batch_size = 120*10,
        num_epochs=10,
        minibatch_size = 256,
        num_sgd_iter = 10,
        gamma = 0.99        
    ).rl_module(
        rl_module_spec = RLModuleSpec(
            module_class = MLPAgent,
            observation_space = cfg_mm_mt5_eval.observation_space,
            action_space = cfg_mm_mt5_eval.action_space,
            model_config = {"dense_layers": [256, 256]}
        )
    ).callbacks(
        on_episode_step = log_multi_agent_episode_metrics,
        callbacks_class=SaveBestMeanEpisode
    )
    .learners(
        num_learners = 1,
        num_gpus_per_learner = 0,
    ).reporting(
        min_train_timesteps_per_iteration = 10,
        min_sample_timesteps_per_iteration = 10,
    ).evaluation(
        evaluation_duration = 1,
        evaluation_duration_unit = "episodes",
        evaluation_interval = 1,
        evaluation_config = AlgorithmConfig.overrides(
            env_config = cfg_mm_mt5_eval.model_dump()
        ),
        evaluation_num_env_runners = 1,
    )
)

save_path = os.path.join(project_path, "checkpoints")
tune.run(
    "PPO",
    config = config,
    stop = {"training_iteration":1000000},
    storage_path = save_path,
    callbacks=[SaveCheckpointData()]
)
