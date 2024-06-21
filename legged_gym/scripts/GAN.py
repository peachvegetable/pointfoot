# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from collections import defaultdict

import numpy as np
import os
from datetime import datetime

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym.models.generator import Generator
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim
import onnxruntime as ort
from legged_gym import LEGGED_GYM_ROOT_DIR


def load_policy(policy_path):
    return ort.InferenceSession(policy_path)


def get_actions(policy, env, cmd, device):
    obs = env.get_observations()
    obs[0][-1] = cmd[-1]
    obs[0][-2] = cmd[-2]
    obs[0][-3] = cmd[-3]
    obs_np = obs.cpu().numpy()
    obs_np = obs_np.reshape(-1)

    inputs = {policy.get_inputs()[0].name: obs_np}
    actions = policy.run(None, inputs)

    action_tensors = torch.tensor(actions[0], dtype=torch.float32, device=device)
    return action_tensors.unsqueeze(0)


def simulate_trajectory(sim_params, policy, env, cmd, device, robot_asset):
    friction = sim_params

    env.update_frictions(friction, robot_asset)
    # make the env updated
    env.step(env.actions)

    actions = get_actions(policy, env, cmd, device)
    obs, _, _, _, _ = env.step(actions)

    return obs.to(device)


def collect_trajectory(sim_traj, step, device):
    tot_traj = []
    for _ in range(step):
        tot_traj.append(sim_traj)
    return torch.stack(tot_traj).to(device)


def categorize_data_by_cmd(data):
    """
    Categorize the data based on the last three items (cmd) in each observation.

    Args:
    - data (torch.Tensor): The input tensor of shape (N, 1, 27), where N is the number of observations.

    Returns:
    - dict: A dictionary where keys are tuples representing cmd values, and values are lists of observations.
    """
    categorized_data = defaultdict(list)

    for obs in data:
        # Extract the last three items as cmd from the last dimension
        cmd = tuple(obs[0, -3:].tolist())  # Convert to tuple to make it hashable
        # Add the observation to the corresponding cmd category
        categorized_data[cmd].append(obs)

    return categorized_data


def train(args, real_data, policy_path):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    env.reset()
    policy = load_policy(policy_path)
    device = ppo_runner.device
    fric_range = env.cfg.domain_rand.friction_range
    real_data = categorize_data_by_cmd(real_data)

    asset_root = os.path.dirname(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    asset_file = os.path.basename(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    asset_options = gymapi.AssetOptions()
    robot_asset = env.gym.load_asset(env.sim, asset_root, asset_file, asset_options)

    criterion = torch.nn.BCELoss()

    # Define discriminator and generator
    discriminator = TransformerDiscriminator(input_dim=27, hidden_dim=80, num_layers=2, output_dim=1).to(device)
    # input_dim = 2 since we only modify friction and added_mass
    # generator = TransformerGenerator(input_dim=1, hidden_dim=40, num_layers=2, output_dim=1).to(device)
    generator = Generator(noise_dim=1, hidden_dim=40, output_range=fric_range).to(device)

    # Define optimizers
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

    # train_cfg.runner.max_iterations
    num_epochs = 10000
    for epoch in range(num_epochs):
        for key in real_data:
            # Extract real data
            real_traj = torch.stack(real_data[key])
            real_traj = real_traj.to(device)
            step = real_traj.shape[0]
            cmd = key

            noise = torch.randn(1).to(device)  # random noise

            # Define labels
            real_label = torch.ones((step, 1)).to(device)
            sim_params = generator(noise)
            fake_label = torch.zeros((step, 1)).to(device)

            # Train discriminator
            disc_optimizer.zero_grad()
            real_output = discriminator(real_traj)
            d_real_loss = criterion(real_output, real_label)
            sim_traj = simulate_trajectory(sim_params, policy, env, cmd, device, robot_asset)
            sim_trajs = collect_trajectory(sim_traj, step, device)
            fake_output = discriminator(sim_trajs)
            d_fake_loss = criterion(fake_output, fake_label)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            disc_optimizer.step()

            # Train generator
            gen_optimizer.zero_grad()
            sim_traj = simulate_trajectory(sim_params, policy, env, cmd, device, robot_asset)
            sim_trajs = collect_trajectory(sim_traj, step, device)
            fake_output = discriminator(sim_trajs)

            g_loss = criterion(fake_output, real_label)
            g_loss.backward()
            gen_optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")


if __name__ == '__main__':
    # Load real data
    real_data_file = '/home/peachvegetable/realdata/rr.npy'
    real_data = real_to_tensor(real_data_file)
    policy_path = '/home/peachvegetable/policy/policy.onnx'
    args = get_args()
    train(args, real_data, policy_path)
