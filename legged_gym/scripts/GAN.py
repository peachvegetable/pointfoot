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
import json
import subprocess
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


# def load_policy(policy_path):
#     return ort.InferenceSession(policy_path)
#
#
# def get_actions(policy, env, device):
#     obs = env.get_observations()
#     obs_np = obs.cpu().numpy()
#     obs_np = obs_np.reshape(-1)
#
#     inputs = {policy.get_inputs()[0].name: obs_np}
#     actions = policy.run(None, inputs)
#
#     action_tensors = torch.tensor(actions[0], dtype=torch.float32, device=device)
#
#     return action_tensors.unsqueeze(0)


# def simulate_trajectory(args, sim_params, policy, cmd, device, step):
#     # Create environment
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     env_handle = env.envs[0]
#     actor_handle = env.actor_handles[0]
#     env.cfg.commands.heading_command = False
#
#     rand_int = random.randint(100, 200)
#
#     friction = sim_params[0]
#     added_mass = sim_params[1]
#     added_com = sim_params[2:]
#
#     env.update_frictions(friction, env_handle, actor_handle)
#     env.update_added_mass_and_base_com(added_mass, added_com, env_handle, actor_handle)
#
#     # Reset env after updating env parameters
#     # env.reset()
#
#     env.update_cmd(cmd)  # obs is re-computed within the update_cmd function
#
#     print(f"friction updated: {friction}, "
#           f"mass updated: {added_mass}, "
#           f"com updated: {added_com.tolist()}")
#
#     tot_traj = []
#     for i in range(step + rand_int):
#         actions = get_actions(policy, env, device)
#         obs, _, _, _, _ = env.step(actions)
#         if i >= rand_int:
#             tot_traj.append(obs)
#
#     tot_traj = torch.stack(tot_traj).to(device)
#
#     # Destroy simulation
#     env.gym.destroy_sim(env.sim)
#     env.gym.destroy_viewer(env.viewer)
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


def train():
    # Load real data
    real_data_file = '/home/peachvegetable/realdata/rr.npy'
    real_data = real_to_tensor(real_data_file)
    device = 'cpu'
    real_data = categorize_data_by_cmd(real_data)
    # print(f"is flat: {env.cfg.terrain.measure_heights_critic is False}")

    output_range = torch.tensor(([0.0, 1.6], [-1., 2.], [-0.03, 0.03], [-0.02, 0.02], [-0.03, 0.03]),
                                device=device)

    criterion = torch.nn.BCELoss()

    # Initialize TensorBoard writer
    log_dir = "./logs/gan_training"
    writer = SummaryWriter(log_dir)

    # Define discriminator and generator
    discriminator = MLPDiscriminator(input_dim=27, hidden_dim=128).to(device)
    generator = MLPGenerator(input_dim=5, hidden_dim=256, output_dim=5, output_range=output_range).to(device)

    # Define optimizers
    disc_lr = 0.0001
    gen_lr = 0.0002
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)

    # num_epochs = train_cfg.runner.max_iterations
    num_epochs = 10000
    for epoch in range(num_epochs):
        # Extract real data
        key = (0.6000000238418579, 0.0, -0.10000000149011612)
        real_traj = torch.stack(real_data[key][1000:2000])
        real_traj = real_traj.to(device)
        step = real_traj.shape[0]
        cmd = key

        noise = torch.randn(5).to(device)  # random noise

        # Define labels
        real_label = torch.ones((step, 1)).to(device)
        sim_params = generator(noise)
        fake_label = torch.zeros((step, 1)).to(device)

        sim_params_path = '/home/peachvegetable/input/sim_params'
        cmd_path = '/home/peachvegetable/input/cmd'
        step_path = '/home/peachvegetable/input/step'

        np.save(sim_params_path, sim_params.cpu().detach().numpy())
        np.save(cmd_path, cmd)
        np.save(step_path, step)

        cmd = 'python simulate_trajectory.py'
        os.system(cmd)

        sim_trajs_path = '/home/peachvegetable/output/sim_traj.npy'
        sim_trajs = np.load(sim_trajs_path, allow_pickle=True)
        for trajs in sim_trajs:
            trajs = torch.tensor(trajs).to('cpu')
        sim_trajs = torch.tensor(sim_trajs).to('cpu')

        # Generate sim_trajs
        # print(f"processing {key} with length {len(real_data[key][1000:2000])}")

        # sim_trajs = simulate_trajectory(args, sim_params, policy, cmd, device, step)

        # Train discriminator
        disc_optimizer.zero_grad()
        real_output = discriminator(real_traj)
        d_real_loss = criterion(real_output, real_label)
        fake_output = discriminator(sim_trajs.detach())
        d_fake_loss = criterion(fake_output, fake_label)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        fake_output = discriminator(sim_trajs.detach())
        g_loss = criterion(fake_output, real_label)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        gen_optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")
        # Save sim_trajs for comparison
        for tensor in sim_trajs:
            tensor = tensor.cpu().numpy()
        file_path = 'sim_trajs/'
        file_name = f"sim_traj{epoch}"
        path = os.path.join(file_path, file_name)
        np.save(path, sim_trajs.cpu().numpy())
        print(f"sim_trajs successfully saved to path: {path}")

        # Save real_trajs
        # for tensor in real_traj:
        #     tensor = tensor.cpu().numpy()
        # file_path = 'real_trajs/'
        # file_name = f"real_traj"
        # path = os.path.join(file_path, file_name)
        # np.save(path, real_traj.cpu().numpy())
        # print(f"real_trajs successfully saved to path: {path}")

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
        writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
        writer.add_scalar('Friction', sim_params[0].item(), epoch)
        writer.add_scalar('Added_mass', sim_params[1].item(), epoch)
        writer.add_scalar('COM/x', sim_params[2].item(), epoch)
        writer.add_scalar('COM/y', sim_params[3].item(), epoch)
        writer.add_scalar('COM/z', sim_params[4].item(), epoch)

    writer.close()

    #     for key in real_data:
    #         # Extract real data
    #         real_traj = torch.stack(real_data[key])
    #         real_traj = real_traj.to(device)
    #         step = real_traj.shape[0]
    #         cmd = key
    #
    #         noise = torch.randn(5).to(device)  # random noise
    #
    #         # Define labels
    #         real_label = torch.ones((step, 1)).to(device)
    #         sim_params = generator(noise)
    #         fake_label = torch.zeros((step, 1)).to(device)
    #
    #         # Generate sim_trajs
    #         print(f"processing {key} with length {len(real_data[key])}")
    #         sim_trajs = simulate_trajectory(sim_params, policy, env, cmd, device, rigid_shape_props, robot_asset, props,
    #                                         base_mass, base_com, env_handle, actor_handle, step)
    #         # Train discriminator
    #         disc_optimizer.zero_grad()
    #         real_output = discriminator(real_traj)
    #         d_real_loss = criterion(real_output, real_label)
    #         fake_output = discriminator(sim_trajs.detach())
    #         d_fake_loss = criterion(fake_output, fake_label)
    #
    #         d_loss = (d_real_loss + d_fake_loss) / 2
    #         d_loss.backward()
    #         disc_optimizer.step()
    #
    #         # Train generator
    #         gen_optimizer.zero_grad()
    #         fake_output = discriminator(sim_trajs.detach())
    #         g_loss = criterion(fake_output, real_label)
    #         g_loss.backward()
    #         gen_optimizer.step()
    #
    #         print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")
    #         # Save sim_trajs for comparison
    #         # if step == 485:
    #         #     for tensor in sim_trajs:
    #         #         tensor = tensor.cpu().numpy()
    #         #     file_path = 'sim_trajs/'
    #         #     file_name = f"sim_traj{epoch}"
    #         #     path = os.path.join(file_path, file_name)
    #         #     np.save(path, sim_trajs.cpu().numpy())
    #         #     print(f"sim_trajs successfully saved to path: {path}")
    #
    #         # Log the losses to TensorBoard
    #         writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
    #         writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
    #         writer.add_scalar('Friction', sim_params[0][0].item(), epoch)
    #         writer.add_scalar('Added_mass', sim_params[0][1].item(), epoch)
    #         writer.add_scalar('COM/x', sim_params[0][2].item(), epoch)
    #         writer.add_scalar('COM/y', sim_params[0][3].item(), epoch)
    #         writer.add_scalar('COM/z', sim_params[0][4].item(), epoch)
    #
    # writer.close()


if __name__ == '__main__':
    train()
