import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from rbc_maenv import DedalusRBC_Env


def env_creator():
    env = DedalusRBC_Env(nagents=10)
    return env


if __name__ == "__main__": 
    ray.init()
    env_name = "ma_rbc"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))  
    
    config = (
            PPOConfig()
            .environment(env=env_name)
            .rollouts(
                num_rollout_workers=4, 
                rollout_fragment_length=64,
                batch_mode="complete_episodes"
            )
            .training(
                train_batch_size=256,
                lr=1e-4,
                gamma=0.99,
                lambda_=0.9,
                use_gae=False,
                clip_param=0.4,
                grad_clip=None,
                entropy_coeff=0.1,
                vf_loss_coeff=0.25,
                sgd_minibatch_size=64,
                num_sgd_iter=10,
                model={"fcnet_hiddens": [512, 512]}
            )
            .debugging(log_level="ERROR")
            .framework(framework="torch")
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )


    tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": 90000},
            checkpoint_freq=10,
            local_dir="~/ray_results/" + env_name,
            config=config.to_dict(),
    )
