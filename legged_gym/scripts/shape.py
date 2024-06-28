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
    policy_path = '/home/peachvegetable/policy/policy.onnx'
    policy = load_policy(policy_path)

    # Test obs
    # sim_traj = torch.tensor([[0.0029,  0.4911,  0.1620,  0.1737, -0.0261, -0.9845,  0.0193, -0.0212,
    #                           0.2523,  0.0188, -0.1258, -0.2884,  0.0294, -0.0191,  0.1346,  0.0556,
    #                           -0.0225,  0.3805,  0.8070,  0.2407, -1.1800,  0.1364, -0.6292, -0.0887,
    #                           0.5000,  0.0000, 0.5000]])
    # cmd = [0.5, 0, 0.5]
    # obs_np = sim_traj.cpu().numpy()
    # obs_np = obs_np.reshape(-1)
    # inputs = {policy.get_inputs()[0].name: obs_np}
    # actions = policy.run(None, inputs)
    # action_tensors = torch.tensor(actions[0], dtype=torch.float32, device=device).unsqueeze(0)
    # env.proprioceptive_obs_buf = sim_traj
    # for _ in range(1000):
    #     env.update_cmd(cmd)
    #     env.step(action_tensors)
    #     print(f"stepping completed")

    # step = 5
    # trajs = collect_trajectory(sim_traj, step, env, policy, cmd, device)
    # print(trajs)
    # actions = torch.tensor(([[1000.0000,  10000.0000, -10000.0000, -100000.0000, -10000.0000, -1000.0000]])).to(device)
    # env.proprioceptive_obs_buf[0][18] = 100.0000
    # env.proprioceptive_obs_buf[0][19] = 100.0000
    # env.proprioceptive_obs_buf[0][20] = -100.0000
    # env.proprioceptive_obs_buf[0][21] = -100.0000
    # env.proprioceptive_obs_buf[0][22] = -100.0000
    # env.proprioceptive_obs_buf[0][23] = -100.0000
    # env.proprioceptive_obs_buf[0][24] = 0.1000
    # env.proprioceptive_obs_buf[0][25] = 0.0000
    # env.proprioceptive_obs_buf[0][26] = 0.2000
    # obs, _, _, _, _ = env.step(actions)
    # print(obs)
    # print(env.actions)
    # obs = torch.tensor(([[-9.4784032e+07, -1.4221987e+08, 8.8256784e+07, 9.6456146e-01,
    #                       -1.5632755e-01, -2.1256208e-01, 0.0000000e+00, 0.0000000e+00,
    #                       0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                       -3.3344738e+06, -2.9579914e+07, 1.7981773e+04, 9.8026720e+06,
    #                       2.8219972e+07, -2.7027070e+06, -1.0000000e+02, -1.0000000e+02,
    #                       -1.0000000e+02, 1.0000000e+02, -1.0000000e+02, -1.0000000e+02,
    #                       3.0000001e-01, 0.0000000e+00, 1.0000000e-01]]))
    # obs_np = obs.cpu().numpy()
    # obs_np = obs_np.reshape(-1)
    # inputs = {policy.get_inputs()[0].name: obs_np}
    # actions = policy.run(None, inputs)
    # print(actions)


    # Test update added_mass
    # base_mass = props[0].mass
    # print(base_mass)
    # print(base_mass.shape)
    # for i in range(5):
    #     obs1 = env.get_observations()
    #     print(f"mass before: {env.base_mass}")
    #     env.update_added_mass(base_mass, i, props, env_handle, actor_handle)
    #     env.compute_observations()
    #     obs2 = env.get_observations()
    #     print(f"mass after: {env.base_mass}, obs changed?: {not torch.equal(obs1, obs2)}")

    # Test update base_com
    print(f"x before: {props[0].com.x}, y before: {props[0].com.y}, z before: {props[0].com.z}")
    base_com = props[0].com
    # print(base_com)
    # print(base_com.shape)
    for i in range(5):
        env.update_base_com(props, base_com, [0.1, 0.1, 0.1], env_handle, actor_handle)
        print(f"x after: {props[0].com.x}, y after: {props[0].com.y}, z after: {props[0].com.z}")
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
    # print(real_data)

    # Test load_policy and get_actions
    # actions = get_actions(policy, env, device=torch.device("cuda"))
    # print(actions.shape)
    # print(actions)
    # print(f"env actions: {env.actions}")

    # Test update_frictions
    # for i in range(100):
    #     env.update_frictions(0.1, robot_asset)
    #     print(f"{i}th update completed")

    # Test generator
    # fric_range = [0, 1]
    # added_mass_range = [0, 2]
    # com_range = [-1, 1]
    # output_range = torch.tensor((fric_range, added_mass_range, com_range, com_range, com_range))
    # generator = TransformerGenerator(input_dim=5, hidden_dim=80, output_dim=5, output_range=output_range).to(device)
    # noise = torch.randn(5).to(device)
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
