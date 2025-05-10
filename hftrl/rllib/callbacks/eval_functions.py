import pdb

from typing import Tuple

from ray.tune.result import TRAINING_ITERATION
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.examples.envs.classes.simple_corridor import SimpleCorridor
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import get_trainable_cls


def custom_eval_function(
    algorithm: Algorithm,
    eval_workers: EnvRunnerGroup,
) -> Tuple[ResultDict, int, int]:
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation EnvRunnerGroup.

    Returns:
        metrics: Evaluation metrics dict.
    """
    pdb.set_trace()
