from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import pdb
import pytz
from ray.tune.registry import get_trainable_cls
from hftrl.env.config.simple_configs import ConfigSimpleEnv
from hftrl.env.physics.simple import SimpleEnv
from ray.tune.registry import register_env

cfg_simple_env = ConfigSimpleEnv(
    action_space = spaces.Dict({
        "market_order":Discrete(3),
        "size": Discrete(2)
        }),
    observation_space = spaces.Dict({
            "volume_profile": Box(low = -np.inf, high = np.inf,
                              shape = (21,), dtype = np.float32)
            }),
    trading_days = [
        (datetime(year=2024, month=3, day=31, hour = 22, minute = 0, tzinfo = pytz.utc),
         datetime(year=2024, month=3, day=31, hour = 22, minute = 20, tzinfo = pytz.utc),
         datetime(year=2024, month=3, day=31, hour = 23, minute=1, tzinfo = pytz.utc))
        ],
    commission_cfg={},
    redis_host = '192.168.1.48',
    redis_port = 6379,
    tick_decimal = 1e9,
    ticker = 'UB',
    suffix_ticker = 'zadd',
    max_b_size = 1,
    max_s_size = 1
    )

register_env("SimpleEnv", lambda config: SimpleEnv(config))

config = (
    get_trainable_cls("PPO").
    get_default_config().
    environment(
        SimpleEnv,
        env_config = cfg_simple_env.dict(),
        render_env = True).
    framework('torch').
    rollouts(num_rollout_workers = 1).
    resources(num_gpus = 0)
)



stop = {
        "training_iteration":1000,
        "timesteps_total":100,
        "episode_reward_mean":100
        }

config.lr = 1e-4
algo = config.build()

for idx in range(stop['training_iteration']):
    result = algo.train()
    print("idx: ",idx, " mean: ", result['env_runners']['episode_reward_mean'])
algo.stop()

print('Finished training')