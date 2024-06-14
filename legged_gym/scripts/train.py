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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.models.discriminator import Discriminator
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim


# Load real data
real_data_file = '/home/peachvegetable/realdata/real_data.npy'
real_data = real_to_tensor(real_data_file)

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    discriminator = Discriminator(input_dim=env.num_obs)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    num_epochs = train_cfg.runner.max_iterations
    for epoch in range(num_epochs):
        # PPO Training Step
        ppo_runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)

        # GAN Training Step
        for real_entry in real_data:
            # Extract real data
            real_input = real_data
            real_label = torch.ones(1)

            # Generate simulated data
            sim_state = env.reset()
            sim_next_state, _ = ppo_runner.alg.actor_critic.act_inference(torch.tensor(sim_state, dtype=torch.float32))
            sim_label = torch.zeros(1)
            sim_input = torch.tensor(sim_state, dtype=torch.float32)

            # Discriminator Training
            disc_optimizer.zero_grad()
            real_output = discriminator(real_input)
            sim_output = discriminator(sim_input)
            real_loss = torch.nn.BCELoss()(real_output, real_label)
            sim_loss = torch.nn.BCELoss()(sim_output, sim_label)
            disc_loss = real_loss + sim_loss
            disc_loss.backward()
            disc_optimizer.step()

            # Update PPO (Generator) using discriminator feedback
            sim_reward = -torch.log(1 - sim_output).detach()
            ppo_runner.alg.actor_critic.update_policy(sim_state, action, sim_reward, sim_next_state)


if __name__ == '__main__':
    args = get_args()
    train(args)
