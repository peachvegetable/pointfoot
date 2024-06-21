from isaacgym import gymapi
import numpy as np

def test_urdf_loading():
    # Initialize Gym
    gym = gymapi.acquire_gym()

    # Configure simulation
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1 / 60.0
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # Use GPU PhysX pipeline
    sim_params.physx.use_gpu = True

    # Create simulator
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    if sim is None:
        raise Exception("Failed to create simulator")

    # Set asset options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True

    # Load URDF
    asset_root = "/home/peachvegetable/limx_rl/pointfoot-legged-gym/resources/robots/PF_P441A/urdf"
    asset_file = "PF_P441A.urdf"

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    if asset is None:
        raise Exception("Failed to load asset")

    print("URDF loaded successfully")

if __name__ == "__main__":
    test_urdf_loading()