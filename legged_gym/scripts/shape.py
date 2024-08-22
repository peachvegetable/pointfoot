from random import random

import numpy as np
import os
from datetime import datetime

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.scripts.GAN import categorize_data_by_cmd
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts import extract_real
import torch.optim as optim

import torch
from legged_gym.utils import get_args, task_registry
# from legged_gym.scripts.GAN import load_policy, get_actions, simulate_trajectory, categorize_data_by_cmd
from legged_gym.models.rough_disc import MLPDiscriminator
from legged_gym.models.generator import TransformerGenerator
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import defaultdict
from legged_gym.scripts.extract_real import real_to_tensor
import random


def main():
    # disc_model_path = '/home/peachvegetable/GAN/output/discriminator'
    # generator_model_path = '/home/peachvegetable/GAN/output/generator'
    # discriminator = MLPDiscriminator(input_dim=27, hidden_dim=128, output_dim=27)
    # discriminator.load_state_dict(torch.load(disc_model_path))

    for i in range(100):
        trajs_path = f'/home/peachvegetable/GAN/output/sim_trajs/sim_traj{i}'
        traj = torch.load(trajs_path)
        traj1 = torch.load(f'/home/peachvegetable/GAN/output/sim_trajs/sim_traj{i+1}')
        print(torch.equal(traj, traj1))
    #
    # command = 'python simulate_trajectory.py'
    # os.system(command)
    # a = f'/home/peachvegetable/GAN/noise0.pt'
    # b = f'/home/peachvegetable/GAN/noise.pt'
    # print(torch.equal(torch.load(a), torch.load(b)))


if __name__ == '__main__':
    # policy_path = '/home/peachvegetable/policy/policy.onnx'
    # device = "cuda"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    # policy = load_policy(policy_path, device)
    main()
