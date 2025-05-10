from ray.rllib.utils.typing import EpisodeType
from ray.rllib.utils.metrics import metrics_logger
import gymnasium as gym
from ray.rllib.core.rl_module import rl_module
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray import tune
from ray.tune.utils import util
import pathlib
import json
import pdb
import os

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class SaveCheckpointData(tune.Callback):
    def on_checkpoint(
        self,
        iteration,
        trials,
        trial,
        checkpoint: tune.Checkpoint,
        **info,
    ):
        """Add extra info to the checkpoint directory"""
        del iteration, trials, info
        checkpoint_path = pathlib.Path(checkpoint.path)
        with open(checkpoint_path / "data.json", "w") as f:
            json.dump(trial.last_result, f, cls=util.SafeFallbackEncoder)


def log_multi_agent_episode_metrics(
    *args,
    episode,
    env_runner,
    metrics_logger,
    env,
    env_index,
    rl_module,
    **kwargs,
):
    """Callback to log performance metrics at the end of an episode."""
    #del args, env_runner, env, env_index, rl_module, kwargs

    if not episode.is_done:
        return
    info = episode._last_added_infos
    if info is None:
        return

    n_last = 1000
    info_done = info['info_done']

    print("the info of metrics is ", info)
    for data_key, data_value in info_done.items():
        metrics_logger.log_value(
            ("metrics", data_key),
            data_value,
            reduce="mean",
            window=n_last
        )

import pdb
class SaveBestMeanEpisode(RLlibCallback):

    def __init__(self):
        super().__init__()

        self.best_mean_reward = -float("inf")
        self.iteration = 0
        self.historical_mean_values = {}

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        self.iteration += 1

        rwd_episode_dict = metrics_logger.stats['env_runners'].get(
            'agent_episode_returns_mean',None
            )
        
        if not rwd_episode_dict:
            return
        if rwd_episode_dict['default_agent'].peek() > self.best_mean_reward:

            self.historical_mean_values[self.iteration] = rwd_episode_dict['default_agent'].peek()

            self.best_mean_reward = rwd_episode_dict['default_agent'].peek() 
            best_agent_checkpoint = os.path.join(
                algorithm.config.env_config['path_render'],
                "checkpoints",
                "best_mean_agent_reward"
                )
            algorithm.save_to_path(best_agent_checkpoint)
            with open(best_agent_checkpoint + "/history.json", "w") as f:
                json.dump(self.historical_mean_values, f)
