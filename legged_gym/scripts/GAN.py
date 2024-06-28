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
from legged_gym.models.generator import TransformerGenerator
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim
import onnxruntime as ort
from legged_gym import LEGGED_GYM_ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
import copy


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


def simulate_trajectory(sim_params, policy, env, cmd, device, rigid_shape_props, robot_asset, body_props,
                        base_mass, base_com, env_handle, actor_handle, step):
    env.cfg.commands.heading_command = False

    friction = sim_params[0][0]
    added_mass = sim_params[0][1]
    added_com = sim_params[0][2:]

    env.update_frictions(friction, rigid_shape_props, robot_asset)
    env.update_added_mass(base_mass, added_mass, body_props, env_handle, actor_handle)
    env.update_base_com(body_props, base_com, added_com, env_handle, actor_handle)

    # Reset env after updating env parameters
    env.reset()

    env.update_cmd(cmd)  # obs is re-computed within the update_cmd function

    tot_traj = []
    for i in range(step):
        actions = get_actions(policy, env, device)
        obs, _, _, _, _ = env.step(actions)
        print(f"Mass updated: {added_mass + base_mass == env.base_mass}")
        print(f"Friction updated: {friction == rigid_shape_props[0].friction}")
        print(f"CMD updated: {env.commands[0][0] == cmd[0] and env.commands[0][1] == cmd[1] and env.commands[0][2] == cmd[2]}")
        print(f"COM updated: {base_com.x + added_com[0] == body_props[0].com.x and base_com.y + added_com[1] == body_props[0].com.y and base_com.z + added_com[2] == body_props[0].com.z}")
        tot_traj.append(obs)

    tot_traj = torch.stack(tot_traj).to(device)
    return tot_traj

# def collect_trajectory(sim_traj, step, env, policy, cmd, device):
#     env.cfg.commands.heading_command = False
#     env.proprioceptive_obs_buf = sim_traj
#     tot_traj = []
#     for i in range(step):
#         # Get actions based on the current sim_traj
#         # Get the next sim_traj based on the generated actions
#         sim_traj, _, _, _, _ = env.step(get_actions(policy, env, cmd, device))
#
#         tot_traj.append(sim_traj)
#
#     tot_traj = torch.stack(tot_traj).to(device)
#
#     return tot_traj


def categorize_data_by_cmd(data):
    """Categorize the data based on the last three items (cmd) in each observation."""
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
    added_mass_range = env.cfg.domain_rand.added_mass_range
    com_range = env.cfg.domain_rand.rand_com_vec
    real_data = categorize_data_by_cmd(real_data)

    asset_root = os.path.dirname(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    asset_file = os.path.basename(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    asset_options = gymapi.AssetOptions()
    robot_asset = env.gym.load_asset(env.sim, asset_root, asset_file, asset_options)
    rigid_shape_props = env.gym.get_asset_rigid_shape_properties(robot_asset)
    env_handle = env.envs[0]
    actor_handle = env.actor_handles[0]
    props = env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

    # Specify output_range
    x, y, z = com_range
    output_range = torch.tensor((fric_range, added_mass_range, [-x, x], [-y, y], [-z, z]),
                                device=device)

    # base mass and base com
    base_mass = copy.copy(props[0].mass)
    base_com = copy.copy(props[0].com)

    criterion = torch.nn.BCELoss()

    # Initialize TensorBoard writer
    log_dir = "./logs/gan_training"
    writer = SummaryWriter(log_dir)

    # Define discriminator and generator
    discriminator = TransformerDiscriminator(input_dim=27, hidden_dim=80, num_layers=2, output_dim=1).to(device)
    generator = TransformerGenerator(input_dim=5, hidden_dim=80, output_dim=5, output_range=output_range).to(device)

    # Define optimizers
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

    # num_epochs = train_cfg.runner.max_iterations
    num_epochs = 2
    for epoch in range(num_epochs):
        for key in real_data:

            # Extract real data
            real_traj = torch.stack(real_data[key])
            real_traj = real_traj.to(device)
            step = real_traj.shape[0]
            cmd = key

            noise = torch.randn(5).to(device)  # random noise

            # Define labels
            real_label = torch.ones((step, 1)).to(device)
            sim_params = generator(noise)
            fake_label = torch.zeros((step, 1)).to(device)

            # Generate sim_trajs
            sim_trajs = simulate_trajectory(sim_params, policy, env, cmd, device, rigid_shape_props, robot_asset, props,
                                            base_mass, base_com, env_handle, actor_handle, step)
            # Train discriminator
            disc_optimizer.zero_grad()
            real_output = discriminator(real_traj)
            d_real_loss = criterion(real_output, real_label)
            fake_output = discriminator(sim_trajs.detach())
            d_fake_loss = criterion(fake_output, fake_label)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            disc_optimizer.step()

            # Train generator
            gen_optimizer.zero_grad()
            fake_output = discriminator(sim_trajs.detach())
            g_loss = criterion(fake_output, real_label)
            g_loss.backward()
            gen_optimizer.step()

            # print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

            # Log the losses to TensorBoard
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
            writer.add_scalar('Friction', sim_params[0][0].item(), epoch)
            writer.add_scalar('Added_mass', sim_params[0][1].item(), epoch)
            writer.add_scalar('COM_x', sim_params[0][2].item(), epoch)
            writer.add_scalar('COM_y', sim_params[0][3].item(), epoch)
            writer.add_scalar('COM_z', sim_params[0][4].item(), epoch)

    writer.close()


if __name__ == '__main__':
    # Load real data
    real_data_file = '/home/peachvegetable/realdata/rr.npy'
    real_data = real_to_tensor(real_data_file)
    policy_path = '/home/peachvegetable/policy/policy.onnx'
    args = get_args()
    train(args, real_data, policy_path)
