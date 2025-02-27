from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import pdb
import pytz
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms import PPOConfig
from hftrl.env.config.simple_configs import ConfigSimpleEnv
from hftrl.env.physics.simple import SimpleEnv
from ray.tune.registry import register_env
from hftrl.env.callbacks.simplecallback import MyCustomCallbacks

import logging
import os

logger = logging.getLogger(__name__)


side_lad = 110
tot = side_lad*2 + 1

project_path = '/home/mora/Documents/projects/HFTRL/hftrl/tests/experimentations/attention_4_contracts/'

cfg_simple_env = ConfigSimpleEnv(
    action_space = spaces.Dict({
        "market_order":Discrete(3),
        "size": Discrete(2)
        }),
    observation_space = spaces.Dict({
            "volume_profile": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "bid_sold": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "ask_bought": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "current_price": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "current_pos": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            # "closed_pos": Box(low = -np.inf, high = np.inf,
            #                   shape = (1,), dtype = np.float32),
            }),
    trading_days = {
        "year": 2024,
        "month_day":{4:[1,2,3,4,5,8,9,10,11,12,15,16]},
        "preload_hour":4,
        "start_hour" : 12,
        "preload_minute": 1,
        "end_hour" : 15,
        "minutes_to_add" : 30},
    commission_cfg={},
    redis_host = '192.168.1.48',
    redis_port = 6379,
    tick_decimal = 1e9,
    ticker = 'UB',
    suffix_ticker = 'zadd',
    max_size = 4,
    side_lad = side_lad,
    enable_render = False, # should be false if it is implemented in the callback, otherwise repeated values
    path_render = project_path
    )

cfg_simple_env_eval = ConfigSimpleEnv(
    action_space = spaces.Dict({
        "market_order":Discrete(3),
        "size": Discrete(2)
        }),
    observation_space = spaces.Dict({
            "volume_profile": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "bid_sold": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "ask_bought": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "current_price": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            "current_pos": Box(low = -np.inf, high = np.inf,
                              shape = (tot,), dtype = np.float32),
            # "closed_pos": Box(low = -np.inf, high = np.inf,
            #                   shape = (1,), dtype = np.float32),
            }),
    trading_days = {
        "year": 2024,
        "month_day":{4:[17,18]},
        "preload_hour":4,
        "start_hour" : 12,
        "preload_minute": 1,
        "end_hour" : 15,
        "minutes_to_add" : 30},
    commission_cfg={},
    redis_host = '192.168.1.48',
    redis_port = 6379,
    tick_decimal = 1e9,
    ticker = 'UB',
    suffix_ticker = 'zadd',
    max_size = 1,
    side_lad = side_lad,
    enable_render = False, # should be false if it is implemented in the callback, otherwise repeated values
    path_render = project_path
    )

register_env("SimpleEnv", lambda config: SimpleEnv(config))

config = (
    PPOConfig().
    resources(
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
        num_gpus= 0
    ).environment(
        env = SimpleEnv,
        env_config = cfg_simple_env.dict()
    ).rollouts(
        num_envs_per_worker = 1,
        num_rollout_workers = 25,
        batch_mode = "truncate_episodes"
    ).training(
        lr=5e-5,
        train_batch_size = 3840,
        sgd_minibatch_size = 256,
        num_sgd_iter = 117,
        model = {"fcnet_hiddens": [256, 256], 
                 "use_attention": True
                 },
        gamma = 0.99
    ).evaluation(
        evaluation_duration = 1,
        evaluation_duration_unit = "episodes",
        evaluation_interval = 2,
        evaluation_config={
            "callbacks": MyCustomCallbacks,
            "env_config": cfg_simple_env_eval.dict()
        }
    ).reporting(
        min_train_timesteps_per_iteration = 1000,
        min_sample_timesteps_per_iteration = 1000,
    ).framework('torch')
    #.callbacks(callbacks_class=MyCustomCallbacks)
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