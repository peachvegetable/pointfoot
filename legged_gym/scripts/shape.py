import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts import extract_real
import torch
import torch.optim as optim

import numpy as np
import torch
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.GAN import load_policy, get_actions, simulate_trajectory
from legged_gym.models.generator import Generator
from legged_gym.models.discriminator import TransformerDiscriminator


def main():
    args = get_args()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    fric_range = env.cfg.domain_rand.friction_range
    print(env.num_obs)
    obs, _ = env.reset()
    device = torch.device("cuda")

    # Test real data
    real_data_file = '/home/peachvegetable/realdata/2024-06-14-10-27-54.npy'
    real_data = extract_real.real_to_tensor(real_data_file)
    real_traj = real_data[0]

    # Test load_policy and get_actions
    policy_path = '/home/peachvegetable/policy/policy.onnx'
    policy = load_policy(policy_path)
    # actions = get_actions(policy, obs)
    # print(actions.shape)
    # print(f"env actions: {env.actions}")
    # print(simulate_trajectory((0.1, 1), policy, env, obs, 1))

    # Test update_frictions
    # env1 = env.get_observations()
    # env.update_frictions(0.15)
    # env.step(env.actions)
    # env2 = env.get_observations()
    # print(torch.equal(env1, env2))

    # Test generator
    generator = Generator(noise_dim=1, hidden_dim=40, output_range=fric_range).to(device)
    noise = torch.randn(1).to(device)
    sim_params = generator(noise)
    tot_traj = []
    for _ in range(10):
        sim_traj = simulate_trajectory(sim_params, policy, env, obs)
        tot_traj.append(sim_traj)
    tot_traj = torch.stack(tot_traj)

    # Test discriminator
    discriminator = TransformerDiscriminator(input_dim=27, hidden_dim=80, num_layers=2, output_dim=1, num_heads=4).to(device)
    sim_output = discriminator(tot_traj)
    real_label = torch.ones((10, 1)).to(device)
    loss = torch.nn.BCELoss()(sim_output, real_label)
    print(loss)


    # Inspect the simulated data structure from env.get_observations()
    # sim_state = env.get_observations()
    # for data in sim_state:
    #     print(type(data))
    #     print(data)
    # print(type(env))
    # print("Simulated Data from env.get_observations():")
    # print(sim_state)
    # print("Type:", type(sim_state))
    # print("Shape:", sim_state.shape)
    # print(env.num_obs)


if __name__ == '__main__':
    # policy_path = '/home/peachvegetable/policy/policy.onnx'
    # device = "cuda"
    # policy = load_policy(policy_path, device)
    main()
