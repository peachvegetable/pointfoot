import rosbag
import numpy as np
import rospy

def process_data(obs_data):
    processed_data = []
    for i in range(len(obs_data)):
        entry = {
            'obs': obs_data[i]['data']
        }
        processed_data.append(entry)
    return processed_data

def read_bag(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    obs_data = []

    for topic, msg, t in bag.read_messages(topics=['/obs_topic']):
        if topic == '/obs_topic':
            obs_data.append({
                'data': msg.data
            })

    bag.close()
    
    # Print data sizes for verification
    print(f"Obs Data: {len(obs_data)} entries")
    
    return obs_data

def save_data(processed_data, filename):
    np.save(filename, processed_data)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    rospy.init_node('read_bag_node')
    bag_file_path = rospy.get_param('~bag_file_path', 'rr1.bag')
    output_file = rospy.get_param('~output_file', 'rr1.npy')

    obs_data = read_bag(bag_file_path)
    processed_data = process_data(obs_data)
    print(f"Processed Data: {len(processed_data)} entries")
    save_data(processed_data, output_file)

