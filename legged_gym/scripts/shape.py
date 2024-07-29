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
    disc_model_path = '/home/peachvegetable/GAN/output/discriminator'
    generator_model_path = '/home/peachvegetable/GAN/output/generator'
    discriminator = MLPDiscriminator(input_dim=27, hidden_dim=128, output_dim=27)
    discriminator.load_state_dict(torch.load(disc_model_path))


if __name__ == '__main__':
    # policy_path = '/home/peachvegetable/policy/policy.onnx'
    # device = "cuda"
    # policy = load_policy(policy_path, device)
    main()
