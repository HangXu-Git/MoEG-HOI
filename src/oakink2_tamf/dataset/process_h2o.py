import time
import numpy as np
import json
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import tqdm

h2o_obj_name = [
    '', 'book', 'espresso', 'lotion', 'spray',
    'milk', 'cocoa', 'chips', 'cappuccino', 
]

def change_from_both_hands_to_single(data_path, save_path):

    data_file = np.load(data_path, allow_pickle=True)
    print(data_file.files)
    data = {key: data_file[key] for key in data_file.files}
    data_file.close()

    data_num = data["x_obj"].shape[0]
    hand_pose_list = []
    hand_shape_list = []
    hand_joint_list = []
    hand_side_list = []
    obj_traj_list = []
    hand_org_list = []
    obj_name_list = []
    action_name_list = []
    nframes_list = []
    info_list = []

    action_set = set()
    
    for index in tqdm.tqdm(range(data_num)):
        for hand_type in ["lhand", "rhand"]:
            if hand_type == "lhand":
                hand_side = "lh"
            else:
                hand_side = "rh"
            info = f"h2o_{index}_{hand_side}"

            is_hand = data[f"is_{hand_type}"][index]
            if is_hand == 0:
                # print(info)
                continue

            hand_pose = data[f"x_{hand_type}"][index]
            hand_shape = data[f"{hand_type}_beta"][index]
            hand_joint = data[f"j_{hand_type}"][index]
            obj_traj = data["x_obj"][index]
            hand_org = data[f"{hand_type}_org"][index]
            object_idx = data["object_idx"][index]
            obj_name = h2o_obj_name[object_idx]
            assert obj_name != ''

            # obj_name = data["obj_name"][index]
            # proc_obj_name = data["proc_obj_name"][index]
            action_name = data["action_name"][index]
            nframes = data["nframes"][index]
            
            hand_pose_list.append(hand_pose)
            hand_shape_list.append(hand_shape)
            hand_joint_list.append(hand_joint)
            hand_side_list.append(hand_side)    
            obj_traj_list.append(obj_traj)
            hand_org_list.append(hand_org)
            obj_name_list.append(obj_name)
            action_name_list.append(action_name)
            nframes_list.append(nframes)
            info_list.append(info)

            action_set.add(action_name)

    print(f"Total actions: {len(action_set)}")
    print(f"Actions: {action_set}")

    print(nframes_list)

    action_set_1 = set()
    for action in action_set:
        print(f"Action: {action}, Primitive: {extract_action_only(action)}")
        action_set_1.add(extract_action_only(action))

    print(f"Total actions after extraction: {len(action_set_1)}")
    print(f"Actions after extraction: {action_set_1}")


    # save_dict = {
    #     "hand_pose": hand_pose_list,
    #     "hand_shape": hand_shape_list,
    #     "hand_joint": hand_joint_list,  
    #     "hand_side": hand_side_list,
    #     "obj_traj": obj_traj_list,
    #     "hand_org": hand_org_list,
    #     "object_name": obj_name_list,
    #     "action_name": action_name_list,
    #     "nframes": nframes_list,
    #     "info": info_list,
    # }
    # with open(save_path, "wb") as ofstream:
    #     pickle.dump(save_dict, ofstream)
            
def extract_action_only(action):
    tokens = action.split(" ")
    lens = len(tokens)
    action_name = " ".join(tokens[:-1]) if lens > 1 else tokens[0]
    
    return action_name

change_from_both_hands_to_single(data_path="data_h2o/h2o/test/data.npz", save_path = "data_h2o/h2o/test/data_single_hand.pkl")  