from collections import defaultdict

import numpy as np
import os

import isaacgym
from legged_gym.envs import *
import torch
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym.models.generator import TransformerGenerator
from legged_gym.models.rough_gen import MLPGenerator
from legged_gym.models.critic import MLPCritic
from legged_gym.scripts.extract_real import real_to_tensor
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import torch.autograd as autograd


def categorize_data_by_cmd(data):
    """Categorize the data based on the last three items (cmd) in each observation."""
    categorized_data = defaultdict(list)

    for obs in data:
        # Extract the last three items as cmd from the last dimension
        cmd = tuple(obs[0, -3:].tolist())  # Convert to tuple to make it hashable
        # Add the observation to the corresponding cmd category
        categorized_data[cmd].append(obs)

    return categorized_data


def wasserstein_loss(fake_output, real_output):
    return torch.mean(fake_output) - torch.mean(real_output)


def gradient_penalty(critic, real_data, fake_data):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to('cpu')
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)

    interpolated_output = critic(interpolated)
    gradients = autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


def train():
    # Load real data
    real_data_file = '/home/peachvegetable/GAN/realdata/rr.npy'
    real_data = real_to_tensor(real_data_file)
    device = 'cpu'
    real_data = categorize_data_by_cmd(real_data)

    output_range = torch.tensor(([0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [-1., 2.],
                                 [-0.03, 0.03], [-0.02, 0.02], [-0.03, 0.03]), device=device)

    # Initialize TensorBoard writer
    log_dir = "./logs/gan_training"
    writer = SummaryWriter(log_dir)

    # Define critic and generator
    critic = MLPCritic(input_dim=27, hidden_dim=128, output_dim=1).to(device)
    generator = MLPGenerator(input_dim=10, hidden_dim=128, output_dim=10, output_range=output_range).to(device)

    # Define optimizers
    disc_lr = 0.001
    gen_lr = 0.004
    critic_optimizer = optim.Adam(critic.parameters(), lr=disc_lr)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)

    num_epochs = 10000
    lambda_gp = 10

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

                cmd_path = '/home/peachvegetable/GAN/input/cmd'
                step_path = '/home/peachvegetable/GAN/input/step'

                np.save(cmd_path, cmd)
                np.save(step_path, step)

                # Train generator
                gen_optimizer.zero_grad()
                noise = torch.tensor(np.random.normal(0, 1, (10)), dtype=torch.float32).to(device)  # random noise
                sim_params = generator(noise)
                sim_params_path = '/home/peachvegetable/GAN/input/sim_params'
                np.save(sim_params_path, sim_params.cpu().detach().numpy())

                cmd = 'python simulate_trajectory.py'
                os.system(cmd)

                sim_trajs_path = '/home/peachvegetable/GAN/output/sim_traj.npy'
                sim_trajs = np.load(sim_trajs_path, allow_pickle=True)
                sim_trajs = torch.from_numpy(sim_trajs).to('cpu')

                fake_output = critic(sim_trajs)
                g_loss = -torch.mean(fake_output)
                g_loss.backward()
                gen_optimizer.step()

                # Train critic
                critic_optimizer.zero_grad()
                real_output = critic(real_traj)
                fake_output = critic(sim_trajs.detach())

                gp = gradient_penalty(critic, real_traj, sim_trajs)
                d_loss = wasserstein_loss(fake_output, real_output) + lambda_gp * gp
                d_loss.backward()
                critic_optimizer.step()

                # Save sim_trajs for comparison
                file_path = 'sim_trajs/'
                file_name = f"sim_traj{epoch}"
                path = os.path.join(file_path, file_name)
                np.save(path, sim_trajs.cpu().numpy())
                print(f"Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, "
                      f"Processing range: {random_2} to {random_2 + random_1}, "
                      f"Processing cmd: {cmd}")

                # Log the losses to TensorBoard
                writer.add_scalar('Loss/Critic', d_loss.item(), epoch)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
                writer.add_scalar('Friction/0', sim_params[0].item(), epoch)
                writer.add_scalar('Friction/1', sim_params[1].item(), epoch)
                writer.add_scalar('Friction/2', sim_params[2].item(), epoch)
                writer.add_scalar('Friction/3', sim_params[3].item(), epoch)
                writer.add_scalar('Friction/4', sim_params[4].item(), epoch)
                writer.add_scalar('Friction/5', sim_params[5].item(), epoch)
                writer.add_scalar('Added_mass', sim_params[6].item(), epoch)
                writer.add_scalar('COM/x', sim_params[7].item(), epoch)
                writer.add_scalar('COM/y', sim_params[8].item(), epoch)
                writer.add_scalar('COM/z', sim_params[9].item(), epoch)

            writer.close()

            disc_model_path = '/home/peachvegetable/GAN/output/critic'
            generator_model_path = '/home/peachvegetable/GAN/output/generator'
            torch.save(critic.state_dict(), disc_model_path)
            torch.save(generator.state_dict(), generator_model_path)


if __name__ == '__main__':
    train()
