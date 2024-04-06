#!/usr/bin/env python
import sys
import gymnasium as gym
import torch
import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rbc_env import DedalusRBC_Env


def main(argv):
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    env = make_vec_env(DedalusRBC_Env, n_envs=30, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=256, policy_kwargs=dict(net_arch=[512, 512]), learning_rate=1e-4, ent_coef=0.01)

    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    model.save("ppo_rbc")

if __name__ == "__main__":
    main(sys.argv[1:])                                                                                                                                                            
