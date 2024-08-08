import numpy as np
import matplotlib.pyplot as plt


# Check if the data is not empty and has the correct shape
def main():
    num_obs = 27
    epoch = 49999
    path = f'/home/peachvegetable/limx_rl/pointfoot-legged-gym/legged_gym/scripts/sim_trajs/sim_traj{epoch}.npy'
    # path = '/home/peachvegetable/GAN/output/sim_traj.npy'
    sim_traj = np.load(path, allow_pickle=True)
    real_path = f'/home/peachvegetable/limx_rl/pointfoot-legged-gym/legged_gym/scripts/real_trajs/real_traj{epoch}.npy'
    real = np.load(real_path, allow_pickle=True)

    step = sim_traj.shape[0]

    print(f"sim_traj shape: {sim_traj.shape}, real_traj shape: {real.shape}")

    # Plot real
    # cmds = [128, 145, 203, 186, 3, 485, 65, 2, 344, 1, 6, 195]
    # for obs in range(num_obs):
    #     real_obs = []
    #     for i in range(step):
    #         real_obs.append(real1[i][0][obs])
    #
    #     real_obs = np.array(real_obs)
    #
    #     # Plot the scalar values
    #     plt.figure(figsize=(10, 6))
    #
    #     # Plot real1 scalar values with a line plot
    #     plt.plot(real_obs, label='real_obs', marker='x', linestyle='--', color='r')
    #
    #     # Add labels, title, and legend
    #     plt.xlabel('Index')
    #     plt.ylabel('Value')
    #     plt.title('Comparison of obs1 and real1 Scalar Values')
    #     plt.legend()
    #
    #     # Show the plot
    #     plt.show()

    # plt.figure(figsize=(10, 6))

    # Plot sim vs real
    for obs in range(num_obs):
        sim_obs = []
        real_obs = []
        for i in range(step):
            sim_obs.append(sim_traj[i][obs])
            real_obs.append(real[i][obs])

        sim_obs = np.array(sim_obs)
        real_obs = np.array(real_obs)

        # Plot the scalar values
        plt.figure(figsize=(10, 6))

        # Plot obs1 scalar values with a line plot
        plt.plot(sim_obs, label='sim_obs', marker='o', linestyle='-', color='b')

        # Plot real1 scalar values with a line plot
        plt.plot(real_obs, label='real_obs', marker='x', linestyle='--', color='r')

        # Add labels, title, and legend
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Comparison of obs1 and real1 Scalar Values')
        plt.legend()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    main()

