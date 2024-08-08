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

import isaacgym
from legged_gym.envs import *
import torch
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym.models.generator import TransformerGenerator
from legged_gym.models.rough_gen import MLPGenerator
from legged_gym.models.rough_disc import MLPDiscriminator
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random


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
    real_data_file = '/home/peachvegetable/GAN/realdata/rr1.npy'
    real_data = real_to_tensor(real_data_file)
    device = 'cpu'
    real_data = categorize_data_by_cmd(real_data)

    # output_range = torch.tensor(([0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [-1., 2.],
    #                              [-0.03, 0.03], [-0.02, 0.02], [-0.03, 0.03]), device=device)

    output_range_fric = torch.tensor(([0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2]),
                                     device=device)
    output_range_mass = torch.tensor(([[-1., 2.]]), device=device)
    output_range_com = torch.tensor(([-0.03, 0.03], [-0.02, 0.02], [-0.03, 0.03]), device=device)

    criterion = torch.nn.BCELoss()

    # Initialize TensorBoard writer
    log_dir = "./logs/gan_training"
    writer = SummaryWriter(log_dir)

    discriminator = MLPDiscriminator(input_dim=27, hidden_dim=128, output_dim=27).to(device)
    generator_fric = MLPGenerator(input_dim=6, hidden_dim=256, output_dim=6, output_range=output_range_fric).to(device)
    generator_mass = MLPGenerator(input_dim=1, hidden_dim=256, output_dim=1, output_range=output_range_mass).to(device)
    generator_com = MLPGenerator(input_dim=3, hidden_dim=256, output_dim=3, output_range=output_range_com).to(device)

    # Define optimizers
    disc_lr = 0.001
    gen_lr = 0.001
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)
    gen_fric_optimizer = optim.Adam(generator_fric.parameters(), lr=gen_lr)
    gen_mass_optimizer = optim.Adam(generator_mass.parameters(), lr=gen_lr)
    gen_com_optimizer = optim.Adam(generator_com.parameters(), lr=gen_lr)

    num_epochs = 500
    for epoch in range(num_epochs):
        for key in real_data:
            length = len(real_data[key])
            if length > 1100:
                random_1 = random.randint(500, 1000)
                random_2 = random.randint(0, length - random_1)
                real_traj = torch.stack(real_data[key][random_2:random_2 + random_1])
                real_traj = real_traj.to(device)
                step = real_traj.shape[0]
                cmd = key

                real_traj = real_traj.reshape(step, 27)

                cmd_path = '/home/peachvegetable/GAN/input/cmd'
                step_path = '/home/peachvegetable/GAN/input/step'

                np.save(cmd_path, cmd)
                np.save(step_path, step)

                # Define labels
                real_label = torch.ones(27).to(device)
                fake_label = torch.zeros(27).to(device)

                noise_fric = torch.tensor(np.random.normal(0, 1, (6)), dtype=torch.float32).to(device)
                noise_mass = torch.tensor(np.random.normal(0, 1, (1)), dtype=torch.float32).to(device)
                noise_com = torch.tensor(np.random.normal(0, 1, (3)), dtype=torch.float32).to(device)

                # Generate sim_params
                sim_params_fric = generator_fric(noise_fric)
                sim_params_mass = generator_mass(noise_mass)
                sim_params_com = generator_com(noise_com)
                fric_path = '/home/peachvegetable/GAN/input/sim_params_fric.pt'
                mass_path = '/home/peachvegetable/GAN/input/sim_params_mass.pt'
                com_path = '/home/peachvegetable/GAN/input/sim_params_com.pt'
                torch.save(sim_params_fric, fric_path)
                np.save(sim_params_mass, mass_path)
                np.save(sim_params_com, com_path)

                # Generate sim_trajs
                command = 'python simulate_trajectory.py'
                os.system(command)

                # Acquire sim_trajs
                sim_trajs_path = '/home/peachvegetable/GAN/output/sim_traj.pt'
                sim_trajs = torch.load(sim_trajs_path, device=device)

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
                gen_fric_optimizer.zero_grad()
                gen_mass_optimizer.zero_grad()
                gen_com_optimizer.zero_grad()
                fake_output = discriminator(sim_trajs)
                g_loss = criterion(fake_output, real_label)
                g_loss.backward()
                gen_fric_optimizer.step()
                gen_mass_optimizer.step()
                gen_com_optimizer.step()

                # # Print gradients to debug
                # for param in generator_fric.parameters():
                #     if param.grad is not None:
                #         print(f"Generator Fric Gradients: {param.grad.mean().item()}")
                #     else:
                #         print("Generator Fric Gradient is None")
                #
                # for param in generator_mass.parameters():
                #     if param.grad is not None:
                #         print(f"Generator Mass Gradients: {param.grad.mean().item()}")
                #     else:
                #         print("Generator Mass Gradient is None")
                #
                # for param in generator_com.parameters():
                #     if param.grad is not None:
                #         print(f"Generator COM Gradients: {param.grad.mean().item()}")
                #     else:
                #         print("Generator COM Gradient is None")
                #
                # for param in discriminator.parameters():
                #     if param.grad is not None:
                #         print(f"Discriminator Gradients: {param.grad.mean().item()}")
                #     else:
                #         print("Discriminator Gradient is None")

                # # Save sim_trajs for comparison
                # file_path = 'sim_trajs/'
                # file_name = f"sim_traj{epoch}"
                # path = os.path.join(file_path, file_name)
                # np.save(path, sim_trajs.cpu().numpy())
                # print(f"Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, "
                #       f"Processing range: {random_2} to {random_2 + random_1}, "
                #       f"Processing cmd: {cmd}")
                #
                # # Save real_trajs
                # file_path = 'real_trajs/'
                # file_name = f"real_traj{epoch}"
                # path = os.path.join(file_path, file_name)
                # np.save(path, real_traj.cpu().detach().numpy())
                # print(f"real_trajs successfully saved to path: {path}")
                #
                # # Save discriminator and generator
                # disc_model_path = '/home/peachvegetable/GAN/output/discriminator'
                # generator_model_path = '/home/peachvegetable/GAN/output/generator'
                # torch.save(discriminator.state_dict(), disc_model_path)
                # torch.save(generator.state_dict(), generator_model_path)

                # Log the losses to TensorBoard
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
                writer.add_scalar('Output/Real', torch.mean(real_output), epoch)
                writer.add_scalar('Output/Fake', torch.mean(fake_output), epoch)
                writer.add_scalar('Friction/0', sim_params_fric[0].item(), epoch)
                writer.add_scalar('Friction/1', sim_params_fric[1].item(), epoch)
                writer.add_scalar('Friction/2', sim_params_fric[2].item(), epoch)
                writer.add_scalar('Friction/3', sim_params_fric[3].item(), epoch)
                writer.add_scalar('Friction/4', sim_params_fric[4].item(), epoch)
                writer.add_scalar('Friction/5', sim_params_fric[5].item(), epoch)
                writer.add_scalar('Added_mass', sim_params_mass.item(), epoch)
                writer.add_scalar('COM/x', sim_params_com[0].item(), epoch)
                writer.add_scalar('COM/y', sim_params_com[1].item(), epoch)
                writer.add_scalar('COM/z', sim_params_com[2].item(), epoch)
    writer.close()


if __name__ == '__main__':
    train()
