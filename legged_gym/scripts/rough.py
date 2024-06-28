from collections import defaultdict

import numpy as np
import os
from datetime import datetime

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.models.rough_disc import TransformerDiscriminator
from legged_gym.models.rough_gen import TransformerGenerator
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim
import onnxruntime as ort
from legged_gym import LEGGED_GYM_ROOT_DIR
from torch.utils.tensorboard import SummaryWriter


def train(args, real_data):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    env.reset()
    device = ppo_runner.device

    criterion = torch.nn.BCELoss()

    # Initialize TensorBoard writer
    log_dir = "./logs/gan_training"
    writer = SummaryWriter(log_dir)

    # Define discriminator and generator
    discriminator = TransformerDiscriminator(input_dim=27, hidden_dim=80, num_layers=2, output_dim=1).to(device)
    generator = TransformerGenerator(input_dim=27, hidden_dim=80, output_dim=27).to(device)

    # Define optimizers
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

    # num_epochs = train_cfg.runner.max_iterations
    num_epochs = 100000
    for epoch in range(num_epochs):
        for data in real_data:
            # Extract real data
            real_traj = data.to(device)

            noise = torch.randn((1, 27)).to(device)  # random noise

            # Define labels
            real_label = torch.ones((1, 1)).to(device)
            sim_traj = generator(noise)
            fake_label = torch.zeros((1, 1)).to(device)

            # Train discriminator
            disc_optimizer.zero_grad()
            real_output = discriminator(real_traj)
            d_real_loss = criterion(real_output, real_label)
            fake_output = discriminator(sim_traj.detach())
            d_fake_loss = criterion(fake_output, fake_label)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            disc_optimizer.step()

            # Train generator
            gen_optimizer.zero_grad()
            fake_output = discriminator(sim_traj)

            g_loss = criterion(fake_output, real_label)
            g_loss.backward(retain_graph=True)
            gen_optimizer.step()

            # print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

            # Log the losses to TensorBoard
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch)

    writer.close()


if __name__ == '__main__':
    # Load real data
    real_data_file = '/home/peachvegetable/realdata/rr.npy'
    real_data = real_to_tensor(real_data_file)
    args = get_args()
    train(args, real_data)

