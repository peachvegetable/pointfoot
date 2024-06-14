import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import torch.optim as optim

import numpy as np
import torch
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # Inspect the simulated data structure from env.reset()
    sim_state, _ = env.reset()
    print(type(env))
    print("Simulated Data from env.reset():")
    print(sim_state)
    print("Type:", type(sim_state))
    print("Shape:", sim_state.shape)


if __name__ == '__main__':
    main()
