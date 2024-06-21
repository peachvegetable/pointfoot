import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def extract_real_features(real_entry):
    # Concatenate all relevant features from the real entry to match the simulated data
    features = np.concatenate([
        real_entry['imu_gyro'],           # Gyroscope data from IMU
        real_entry['proj_gracity'],  # Projected gravity calculated using IMUData
        real_entry['joint_positions'],    # Joint positions from Joint States
        real_entry['joint_velocities'],   # Joint velocities from Joint States
        real_entry['action_data'],   # Actions from policy
        real_entry['joint_torques']
    ])
    return features


def real_to_tensor(real_data_file, step):
    # Load real data
    real_data = np.load(real_data_file, allow_pickle=True)

    # Convert real data to torch.Tensor and verify the shape
    output = []
    real_data_tensor = []
    for real_entry in real_data:
        real_state = extract_real_features(real_entry)
        real_tensor = torch.tensor(real_state, dtype=torch.float32)
        if len(real_data_tensor) <= step - 1:
            real_data_tensor.append(real_tensor.unsqueeze(0))
        else:
            real_data_tensor = torch.stack(real_data_tensor)
            output.append(real_data_tensor)
            real_data_tensor = []

    return torch.stack(output[:-1]).to(torch.device("cuda:0"))

