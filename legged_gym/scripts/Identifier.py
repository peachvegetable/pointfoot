from legged_gym.models.LSTM import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train():
    device = 'cpu'
    log_dir = "./logs/identifier_training"
    writer = SummaryWriter(log_dir)
    # output_range = torch.tensor(([0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [0.0, 0.2], [-1., 2.],
    #                              [-0.03, 0.03], [-0.02, 0.02], [-0.03, 0.03]), device=device)
    model = LSTMModel(input_dim=27, hidden_dim=512, output_dim=10)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    step = 1000
    epoch = 10000
    cmd = (0.0, 0.0, 0.0)
    i = 0
    # friction = torch.tensor((0.001, 0., 0., 0., 0., 0.))
    # mass = torch.tensor([0.5])
    # com = torch.tensor((0., 0.001, 0.))
    # cmd = (0.0, 0.0, 0.0)
    while True:
        model.train()
        friction = torch.rand(6) * 0.2
        mass = torch.rand(1) * 3 - 1
        com = torch.rand(3)
        com[0] = com[0] * 0.06 - 0.03
        com[1] = com[1] * 0.04 - 0.02
        com[2] = com[2] * 0.06 - 0.03

        step_path = '/home/peachvegetable/GAN/input/step.npy'
        cmd_path = '/home/peachvegetable/GAN/input/cmd.npy'
        terminate_path = '/home/peachvegetable/GAN/output/terminate.npy'

        friction_path = '/home/peachvegetable/GAN/input/sim_params_fric.pt'
        mass_path = '/home/peachvegetable/GAN/input/sim_params_mass.pt'
        com_path = '/home/peachvegetable/GAN/input/sim_params_com.pt'

        np.save(step_path, step)
        np.save(cmd_path, cmd)
        torch.save(friction, friction_path)
        torch.save(mass, mass_path)
        torch.save(com, com_path)

        # Generate sim_trajs
        command = 'python simulate_trajectory.py'
        os.system(command)

        # Check if the trajectory is complete (i.e. the robot did not fall)
        terminate = np.load(terminate_path)
        print(f'Terminate: {terminate == 1}')
        if terminate:
            continue

        i += 1

        # Acquire sim_trajs
        sim_trajs_path = '/home/peachvegetable/GAN/output/sim_traj.pt'
        sim_trajs = torch.load(sim_trajs_path).to(device)

        real_param = torch.cat((friction, mass, com))

        # Train the model
        optimizer.zero_grad()
        output = model(sim_trajs)
        loss = (criterion(output[0], real_param) + criterion(output[1], real_param) + criterion(output[2], real_param)) / 3
        avg_output = torch.mean(output, dim=0)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{i + 1}/{epoch}], loss: {loss.item()}")

        # Save model
        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), f'/home/peachvegetable/GAN/output/models/model{i}')

        # Save trajs and parameters
        trajs_path = f'/home/peachvegetable/GAN/output/sim_trajs/sim_traj{i}'
        param_path = f'/home/peachvegetable/GAN/output/real_params/params{i}'
        torch.save(sim_trajs, trajs_path)
        torch.save(real_param, param_path)

        writer.add_scalar('Loss', loss.item(), i)
        writer.add_scalars('Friction/0', {'real': friction[0].item(), 'fake': avg_output[0].item()}, i)
        writer.add_scalars('Friction/1', {'real': friction[1].item(), 'fake': avg_output[1].item()}, i)
        writer.add_scalars('Friction/2', {'real': friction[2].item(), 'fake': avg_output[2].item()}, i)
        writer.add_scalars('Friction/3', {'real': friction[3].item(), 'fake': avg_output[3].item()}, i)
        writer.add_scalars('Friction/4', {'real': friction[4].item(), 'fake': avg_output[4].item()}, i)
        writer.add_scalars('Friction/5', {'real': friction[5].item(), 'fake': avg_output[5].item()}, i)
        writer.add_scalars('Added_mass', {'real': mass.item(), 'fake': avg_output[6].item()}, i)
        writer.add_scalars('COM/x', {'real': com[0].item(), 'fake': avg_output[7].item()}, i)
        writer.add_scalars('COM/y', {'real': com[1].item(), 'fake': avg_output[8].item()}, i)
        writer.add_scalars('COM/z', {'real': com[2].item(), 'fake': avg_output[9].item()}, i)

    writer.close()


if __name__ == '__main__':
    train()





