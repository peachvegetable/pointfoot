from collections import defaultdict

import numpy as np
import os

import isaacgym
from legged_gym.envs import *
import torch
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym.models.generator import TransformerGenerator
from legged_gym.models.rrough_gen import MLPGenerator
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
    device = 'cuda'

    criterion = torch.nn.BCELoss()

    # Initialize TensorBoard writer
    log_dir = "./logs/gan_training"
    writer = SummaryWriter(log_dir)

    discriminator = MLPDiscriminator(input_dim=27, hidden_dim=128, output_dim=27).to(device)
    generator = MLPGenerator(input_dim=27, hidden_dims=[256, 512, 256], output_dim=27).to(device)

    # Define optimizers
    disc_lr = 0.001
    gen_lr = 0.001
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)

    length = real_data.shape[0]

    num_epochs = 50000
    for epoch in range(num_epochs):
        random_1 = random.randint(500, 1000)
        random_2 = random.randint(0, length - random_1)
        real_traj = real_data[random_2:random_2 + random_1]
        real_traj = real_traj.to(device)
        step = real_traj.shape[0]

        real_traj = real_traj.reshape(step, 27)

        # Define labels
        real_label = torch.ones(27).to(device)
        fake_label = torch.zeros(27).to(device)

        # Generates output
        noise = torch.randn((step, 27)).to(device)
        output = generator(noise).to(device)

        # Train discriminator
        disc_optimizer.zero_grad()
        real_output = discriminator(real_traj)
        d_real_loss = criterion(real_output, real_label)
        fake_output = discriminator(output.detach())
        d_fake_loss = criterion(fake_output, fake_label)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        fake_output = discriminator(output)
        print(fake_output.grad)
        g_loss = criterion(fake_output, real_label)
        g_loss.backward()
        gen_optimizer.step()

        # print(f"real_output: {real_output}, fake_output: {fake_output}")

        # Print gradients to debug
        for param in generator.parameters():
            if param.grad is not None:
                print(f"Generator Fric Gradients: {param.grad.mean().item()}")
            else:
                print("Generator Fric Gradient is None")

        for param in discriminator.parameters():
            if param.grad is not None:
                print(f"Discriminator Gradients: {param.grad.mean().item()}")
            else:
                print("Discriminator Gradient is None")

        # Save sim_trajs for comparison
        file_path = 'sim_trajs/'
        file_name = f"sim_traj{epoch}"
        path = os.path.join(file_path, file_name)
        np.save(path, output.cpu().detach().numpy())

        # Save real_trajs
        file_path = 'real_trajs/'
        file_name = f"real_traj{epoch}"
        path = os.path.join(file_path, file_name)
        np.save(path, real_traj.cpu().detach().numpy())
        print(f"real_trajs successfully saved to path: {path}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

        disc_model_path = '/home/peachvegetable/GAN/output/discriminator'
        generator_model_path = '/home/peachvegetable/GAN/output/generator'
        torch.save(discriminator.state_dict(), disc_model_path)
        torch.save(generator.state_dict(), generator_model_path)

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
        writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
        writer.add_scalar('Output/Real', torch.mean(real_output), epoch)
        writer.add_scalar('Output/Fake', torch.mean(fake_output), epoch)

    writer.close()


if __name__ == '__main__':
    train()
