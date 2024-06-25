import numpy as np
import os
from datetime import datetime

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts import extract_real
import torch
import torch.optim as optim

import numpy as np
import torch
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.GAN import load_policy, get_actions, simulate_trajectory, categorize_data_by_cmd
from legged_gym.models.generator import TransformerGenerator
from legged_gym.models.discriminator import TransformerDiscriminator
from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import defaultdict



def main():
    args = get_args()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    fric_range = env.cfg.domain_rand.friction_range
    obs, _ = env.reset()
    asset_root = os.path.dirname(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    asset_file = os.path.basename(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    asset_options = gymapi.AssetOptions()
    robot_asset = env.gym.load_asset(env.sim, asset_root, asset_file, asset_options)
    device = ppo_runner.device
    env_handle = env.envs[0]
    actor_handle = env.actor_handles[0]
    props = env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

    # Test obs
    # print(obs.shape)
    # print(obs[0][-3:])

    # Test update added_mass
    # base_mass = props[0].mass
    # print(base_mass)
    # print(base_mass.shape)
    # print(f"mass before: {env.base_mass}")
    # env.update_added_mass(env, base_mass, 1)
    # print(f"mass after: {env.base_mass}")

    # Test update base_com
    # print(f"x before: {props[0].com.x}, y before: {props[0].com.y}, z before: {props[0].com.z}")
    # base_com = props[0].com
    # print(base_com)
    # print(base_com.shape)
    # env.update_base_com(props, base_com, [0.1, 0.1, 0.1], env_handle, actor_handle)
    # print(f"x after: {props[0].com.x}, y after: {props[0].com.y}, z after: {props[0].com.z}")
    print(props[0])

    # Test cmd
    # cmd = [0.5, 0, 0.5]
    # env.update_cmd(cmd)
    # obs1 = env.get_observations()
    # print(torch.equal(obs, obs1))
    # print(f"First obs: {obs}")
    # print(f"Second obs: {obs1}")

    # Test real data
    # real_data_file = '/home/peachvegetable/realdata/rr.npy'
    # real_data = extract_real.real_to_tensor(real_data_file)
    # real_data = categorize_data_by_cmd(real_data)
    # for data in real_data:
    #     print(data)

    # Test load_policy and get_actions
    policy_path = '/home/peachvegetable/policy/policy.onnx'
    policy = load_policy(policy_path)
    # actions = get_actions(policy, env, device=torch.device("cuda"))
    # print(actions.shape)
    # print(actions)
    # print(f"env actions: {env.actions}")

    # Test update_frictions
    # for i in range(100):
    #     env.update_frictions(0.1, robot_asset)
    #     print(f"{i}th update completed")

    # Test generator
    # generator = TransformerGenerator(input_dim=5, hidden_dim=80, output_dim=5).to(device)
    # noise = torch.randn(1, 5).to(device)
    # sim_params = generator(noise)
    # print(sim_params)
    # print(sim_params.shape)

    # Test discriminator
    # discriminator = TransformerDiscriminator(input_dim=27, hidden_dim=80, num_layers=2, output_dim=1, num_heads=4).to(device)
    # sim_output = discriminator(tot_traj)
    # real_label = torch.ones((10, 1)).to(device)
    # loss = torch.nn.BCELoss()(sim_output, real_label)
    # print(loss)


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
