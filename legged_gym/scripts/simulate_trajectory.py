import json
import subprocess
import sys
from collections import defaultdict

import numpy as np
import os
from datetime import datetime

import isaacgym
from isaacgym import gymapi, gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym.models.generator import TransformerGenerator
from legged_gym.models.rough_gen import MLPGenerator
from legged_gym.models.rough_disc import MLPDiscriminator
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim
import onnxruntime as ort
from legged_gym import LEGGED_GYM_ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
import copy
import random


def load_policy(policy_path):
    return ort.InferenceSession(policy_path)


def get_actions(policy, env, device):
    obs = env.get_observations()
    obs_np = obs.cpu().numpy()
    obs_np = obs_np.reshape(-1)

    inputs = {policy.get_inputs()[0].name: obs_np}
    actions = policy.run(None, inputs)

    action_tensors = torch.tensor(actions[0], dtype=torch.float32, device=device)

    return action_tensors.unsqueeze(0)


def simulate_trajectory(args, sim_params_path, policy_path, cmd_path, device, step_path):
    sim_params = np.load(sim_params_path, allow_pickle=True)
    policy = load_policy(policy_path)
    cmd = np.load(cmd_path, allow_pickle=True)
    step = np.load(step_path, allow_pickle=True)

    # Create environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env_handle = env.envs[0]
    actor_handle = env.actor_handles[0]
    env.cfg.commands.heading_command = False

    rand_int = random.randint(100, 200)

    friction = sim_params[0]
    added_mass = sim_params[1]
    added_com = sim_params[2:]

    env.update_frictions(friction, env_handle, actor_handle)
    env.update_added_mass_and_base_com(added_mass, added_com, env_handle, actor_handle)

    # Reset env after updating env parameters
    # env.reset()

    env.update_cmd(cmd)  # obs is re-computed within the update_cmd function

    print(f"friction updated: {friction}, "
          f"mass updated: {added_mass}, "
          f"com updated: {added_com.tolist()}")

    tot_traj = []
    for i in range(step + rand_int):
        actions = get_actions(policy, env, device)
        obs, _, _, _, _ = env.step(actions)
        if i >= rand_int:
            tot_traj.append(obs)

    tot_traj = torch.stack(tot_traj).to(device)

    for tensor in tot_traj:
        tensor = tensor.cpu().numpy()
    file_path = '/home/peachvegetable/output/sim_traj'
    np.save(file_path, tot_traj.cpu().numpy())
    print(f"sim_trajs successfully saved to path: {file_path}")


if "__main__" == __name__:
    args = get_args()
    args.task = 'pointfoot_flat'
    args.headless = 'True'
    device = 'cpu'
    policy_path = '/home/peachvegetable/policy/policy.onnx'
    cmd_path = '/home/peachvegetable/input/cmd.npy'
    sim_params_path = '/home/peachvegetable/input/sim_params.npy'
    step_path = '/home/peachvegetable/input/step.npy'

    simulate_trajectory(args, sim_params_path, policy_path, cmd_path, device, step_path)
