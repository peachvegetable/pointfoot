import rosbag
import numpy as np
from controller_msgs.msg import IMUData, JointState, JointCmd
from gazebo_msgs.msg import ModelStates, LinkStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rospy

def process_data(imu_data, joint_cmds, joint_states, model_states, odometry_data):
    processed_data = []
    for i in range(min(len(imu_data), len(joint_cmds), len(joint_states), len(model_states), len(odometry_data))):
        entry = {
            'imu_quat': imu_data[i]['quat'],
            'imu_acc': imu_data[i]['acc'],
            'imu_gyro': imu_data[i]['gyro'],
            'joint_positions': joint_states[i]['qpos'],
            'joint_velocities': joint_states[i]['qvel'],
            'joint_torques': joint_states[i]['tau'],
            'timestamp': imu_data[i]['timestamp']
        }
        processed_data.append(entry)
    return processed_data

def read_bag(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    imu_data = []
    joint_cmds = []
    joint_states = []
    model_states = []
    odometry_data = []

    for topic, msg, t in bag.read_messages(topics=['/ImuData', '/RobotCmdPointFoot', '/RobotStatePointFoot', '/gazebo/model_states', '/ground_truth/state']):
        if topic == '/ImuData':
            imu_data.append({
                'timestamp': msg.imustamp,
                'status': msg.status,
                'euler': np.array(msg.euler),
                'quat': np.array(msg.quat),
                'acc': np.array(msg.acc),
                'gyro': np.array(msg.gyro)
            })
        elif topic == '/RobotCmdPointFoot':
            joint_cmds.append({
                'timestamp': t.to_sec(),
                'q': np.array(msg.q),
                'v': np.array(msg.v),
                'tau': np.array(msg.tau),
                'kp': np.array(msg.kp),
                'kd': msg.kd,
                'mode': np.array(msg.mode)
            })
        elif topic == '/RobotStatePointFoot':
            joint_states.append({
                'timestamp': t.to_sec(),
                'qpos': np.array(msg.q),
                'qvel': np.array(msg.v),
                'qacc': np.array(msg.vd),
                'tau': np.array(msg.tau),
                'na': msg.na
            })
        elif topic == '/gazebo/model_states':
            model_states.append({
                'timestamp': t.to_sec(),
                'name': msg.name,
                'pose': msg.pose,
                'twist': msg.twist
            })
        elif topic == '/ground_truth/state':
            odometry_data.append({
                'timestamp': t.to_sec(),
                'pose': msg.pose.pose,
                'twist': msg.twist.twist
            })

    bag.close()
    
    # Print data sizes for verification
    print(f"IMU Data: {len(imu_data)} entries")
    print(f"Joint Commands: {len(joint_cmds)} entries")
    print(f"Joint States: {len(joint_states)} entries")
    print(f"Model States: {len(model_states)} entries")
    print(f"Odometry Data: {len(odometry_data)} entries")
    
    return imu_data, joint_cmds, joint_states, model_states, odometry_data

def save_data(processed_data, filename):
    np.save(filename, processed_data)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    rospy.init_node('read_bag_node')
    bag_file_path = rospy.get_param('~bag_file_path', '/home/charles/bagfiles/2024-06-05-10-30-25.bag')
    output_file = rospy.get_param('~output_file', '/home/charles/bagfiles/real_data.npy')

    imu_data, joint_cmds, joint_states, model_states, odometry_data = read_bag(bag_file_path)
    processed_data = process_data(imu_data, joint_cmds, joint_states, model_states, odometry_data)
    save_data(processed_data, output_file)

