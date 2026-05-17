import time
import numpy as np
import json
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
    
class GRAB(Dataset):
    def __init__(
        self, 
        data_path, 
        data_obj_pc_path, 
        text_json, 
        max_nframes=150, 
        data_ratio=1.0, 
        refine_text_path=None,
        augm=False, 
        **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data grab!!!")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

            self.hand_pose = data["hand_pose"]  # shape is (data_num, 238, 99)
            self.hand_shape = data["hand_shape"]  # shape is (data_num, 238, 10)
            self.hand_joint = data["hand_joint"]  # shape is (data_num, 238, 21, 3)
            self.hand_side = data["hand_side"]  # list of ["lh", "rh"], length is data_num
            self.obj_traj = data["obj_traj"]  # shape is (data_num, 238, 9)
            self.hand_org = data["hand_org"]  # shape is (data_num, 238, d)
            self.object_name = data["object_name"]  # list of obj_name, length is data_num
            self.proc_object_name = data["proc_object_name"]  # list of proc_obj_name, length is data_num
            self.action_name = data["action_name"]  # list of action name, length is data_num
            self.nframes = data["nframes"]  # list of nframes,
            self.info = data["info"]  # list of info, length is data_num

        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.refine_text_path = refine_text_path
        if self.refine_text_path is not None:
            with open(self.refine_text_path, "r") as f:
                self.refine_text_dict = json.load(f)
        else:
            self.refine_text_dict = None

        self.object_model = build_object_model(data_obj_pc_path)
        print("Finish to read data grab!!!", f"{time.time()-start_time:.2f}s")

    def __len__(self):
        return int(len(self.action_name)*self.data_ratio)
    
    def __getitem__(self, index):
        info = self.info[index]
        nframes = self.nframes[index]

        # len
        if nframes > self.max_nframes:  # 对于长于max_nframes的样本，随机截取一段
            init_frame = np.random.randint(0, nframes-self.max_nframes)
            nframes = self.max_nframes
        else:
            init_frame = 0

        # obj_traj
        obj_traj = self.obj_traj[index][init_frame:init_frame+self.max_nframes]
        if self.augm:
            obj_traj[:nframes], aug_rotmat, aug_trans = augmentation(obj_traj[:nframes])
        obj_traj = np.expand_dims(obj_traj, axis=0)  # (1, nframes, 10)

        # hand pose
        hand_pose = self.hand_pose[index][init_frame:init_frame+self.max_nframes]
        if self.augm:
            hand_org = self.hand_org[index][init_frame:init_frame+nframes]
            hand_pose[:nframes], _, _ \
                = augmentation(
                    hand_pose[:nframes], 
                    hand_org=hand_org, 
                    aug_rotmat=aug_rotmat, 
                    aug_trans=aug_trans
                )
            
        # hand_joint
        hand_joint = self.hand_joint[index][init_frame:init_frame+self.max_nframes]  # (nframes, 21, 3)
        hand_shape = self.hand_shape[index][init_frame:init_frame+self.max_nframes]  # (nframes, 10)
        hand_side = self.hand_side[index]  # "lh" or "rh"

        # mask
        max_nframes = self.max_nframes
        mask = get_valid_mask(max_nframes, nframes) # max_nframes: 2x frames

        # obj pc
        object_name = self.object_name[index]  # str
        proc_object_name = self.proc_object_name[index]  # str
        obj_list = [object_name]  # list of object name, length is 1
        action_name = self.action_name[index]  # str
        
        text = process_text(
            action_name, 
            proc_object_name, 
            self.text_description, 
            return_key=True
        )
        
        _, obj_pc, obj_pc_normal, _ = self.object_model(object_name)
        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(obj_pc, return_params=True)
        
        obj_pointcloud = np.concatenate([obj_pc, obj_pc_normal], axis=-1)  # (1024, 6)
        obj_pointcloud = np.expand_dims(obj_pointcloud, axis=0) # (1, 1024, 6)  # 和oakink2数据集一致

        if self.refine_text_dict is None:
            hand_level_text = None
            finger_level_text = None
            joint_level_text = None
        else:
            info_name = info.split("_")[0]+"_"+info.split("_")[1]
            segment_refine_text = self.refine_text_dict[info_name][hand_side + '_refine_text']
            hand_level_text = segment_refine_text["hand_level"]
            finger_level_text = segment_refine_text["finger_level"]
            joint_level_text = segment_refine_text["joint_level"]

        res = {
            "info": info,
            "len": nframes,
            "mask": mask,
            "pose_repr": hand_pose,
            "shape": hand_shape,
            "hand_side": hand_side,
            "hand_joint": hand_joint,
            "text": text,
            "obj_list": obj_list,
            "obj_num": len(obj_list),
            "obj_traj": obj_traj,   # (1, nframes, 9)  use obj_num
            "hand_level_text": hand_level_text,
            "finger_level_text": finger_level_text,
            "joint_level_text": joint_level_text,
            "primitive": action_name,
            "obj_pointcloud": obj_pointcloud,   # (1, 1024, 6)  use obj_num
            "normalized_obj_pc": normalized_obj_pc,
            "obj_cent": obj_norm_cent,
            "obj_scale": obj_norm_scale,
        }

        return res


class ObjectModel:
    def __init__(self, pkl_file):
        self.pkl_file = pkl_file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            self.object_name = data["object_name"]
            self.obj_pcs = data["obj_pcs"]  # dict of object name, each value is (1024, 3)
            self.obj_pc_normals = data["obj_pc_normals"]  # dict of object name, each value is (1024, 3)
            self.point_sets = data["point_sets"]  # dict of object name, each value is (1024), 最远点采样的index
            self.obj_path = data["obj_path"]
            if "obj_pc_top" in data:
                self.obj_pc_top = data["obj_pc_top"]  # dict of object name, each value is (1024), 0/1
            else:
                self.obj_pc_top = None

    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        point_set = self.point_sets[object_name].copy()
        obj_pc = self.obj_pcs[object_name].copy()
        obj_pc_normal = self.obj_pc_normals[object_name].copy()
        obj_path = self.obj_path[object_name]
        if self.obj_pc_top is not None:
            obj_pc_top = self.obj_pc_top[object_name].copy()
            return point_set, obj_pc, obj_pc_normal, obj_path, obj_pc_top
        else:
            return point_set, obj_pc, obj_pc_normal, obj_path


def build_object_model(pkl_file):
    object_model = ObjectModel(pkl_file)
    return object_model

# def get_valid_mask(is_lhand, is_rhand, nframes, valid_nframes):
#     valid_mask_lhand = np.zeros((nframes))
#     valid_mask_rhand = np.zeros((nframes))
#     valid_mask_obj = np.zeros((nframes))
#     valid_mask_lhand[:valid_nframes] = 1
#     valid_mask_rhand[:valid_nframes] = 1
#     valid_mask_obj[:valid_nframes] = 1
#     if not is_lhand:
#         valid_mask_lhand[:] = 0
#     if not is_rhand:
#         valid_mask_rhand[:] = 0
#     return (
#         valid_mask_lhand.astype(int), 
#         valid_mask_rhand.astype(int), 
#         valid_mask_obj.astype(int)
#     )

def get_valid_mask(nframes, valid_nframes):
    mask = np.zeros((nframes))
    mask[:valid_nframes] = 1
    return mask.astype(int)

# def process_text(
#     action_name, 
#     object_name, 
#     is_lhand, is_rhand, 
#     text_descriptions, return_key=False, 
# ):
#     if is_lhand and is_rhand:
#         text = f"{action_name} {object_name} with both hands."
#     elif is_lhand:
#         text = f"{action_name} {object_name} with left hand."
#     elif is_rhand:
#         text = f"{action_name} {object_name} with right hand."
#     text_key = text.capitalize()
#     if return_key:
#         return text_key
#     else:
#         text_description = text_descriptions[text_key]
#         text = np.random.choice(text_description)
#         return text

def process_text(
    action_name, 
    object_name, 
    text_descriptions, return_key=False, 
):
    text = f"{action_name} the {object_name}."
    text_key = text.capitalize()
    if return_key:
        return text_key
    else:
        text_description = text_descriptions[text_key]
        text = np.random.choice(text_description)
        return text
    
def get_contact_map(idx, v_num, is_hand):
    contact_map = np.zeros(v_num)
    if is_hand:
        contact_map[idx] = 1
    return contact_map

def pc_normalize(pc, return_params=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / scale
    if return_params:
        return pc, centroid, scale
    else:
        return pc
    
def process_dist_map(
    max_nframes, init_frame, 
    cf_idx, cov_idx, chj_idx, 
    dist_value, is_hand
):
    dist_map = np.zeros((max_nframes, 1024, 21), dtype=np.float32)
    if is_hand:
        f_idx_filtered = np.where((init_frame<=cf_idx) & (cf_idx<init_frame+max_nframes))[0]
        cf_idx_selected = cf_idx[f_idx_filtered]
        cf_idx_moved = cf_idx_selected-init_frame
        cov_idx_selected = cov_idx[f_idx_filtered]
        chj_idx_selected = chj_idx[f_idx_filtered]
        dist_value_selected = dist_value[f_idx_filtered]
        dist_map[cf_idx_moved, cov_idx_selected, chj_idx_selected] = dist_value_selected
    return dist_map


# augmentation function for x_obj and x_hand
def augmentation(
    X, 
    r_rot=[5, 20, 5], r_trans=[0.0, 0.0, 0.0],
    hand_org=None, 
    aug_rotmat=None,
    aug_trans=None,
):
    """
        x: x_obj or x_hand (nframes, 9 or 99)
        r_rot: range of random rotation
        r_trans: range of random translation
    """
    nframes = X.shape[0]
    trans = torch.FloatTensor(X[:, :3])
    rot6d = torch.FloatTensor(X[:, 3:9])

    rotmat = rot6d_to_rotmat(rot6d)
    if hand_org is not None:
        trans += hand_org
    trans = trans.unsqueeze(2)
    extmat = torch.cat([rotmat, trans], dim=2)
    homo = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1)
    homo = homo.expand(nframes, -1, -1)
    extmat = torch.cat([extmat, homo], dim=1)

    if aug_rotmat is None:
        aug_rotmat = get_augm_rot(*r_rot)
    if aug_trans is None:
        aug_trans = get_augm_trans(*r_trans)
    aug_extmat = torch.cat([aug_rotmat, aug_trans], dim=1)
    aug_homo = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
    aug_extmat = torch.cat([aug_extmat, aug_homo], dim=0)

    augmented_extmat = torch.einsum("ij,fjk->fik", aug_extmat, extmat)
    augmented_trans = augmented_extmat[:, :3, 3]
    augmented_rotmat = augmented_extmat[:, :3, :3]
    augmented_rot6d = rotmat_to_rot6d(augmented_rotmat)
    augmented_X = torch.cat([augmented_trans, augmented_rot6d], dim=1)
    augmented_X = augmented_X.numpy()
    X[..., :9] = augmented_X
    if hand_org is not None:
        X[..., :3] -= hand_org
    return X, aug_rotmat, aug_trans

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d

def get_augm_rot(r_x_rot, r_y_rot, r_z_rot):
    if r_x_rot != 0:
        x_angle = np.random.randint(-r_x_rot, r_x_rot)
    else:
        x_angle = 0
    if r_y_rot != 0:
        y_angle = np.random.randint(-r_y_rot, r_y_rot)
    else:
        y_angle = 0
    if r_z_rot != 0:
        z_angle = np.random.randint(-r_z_rot, r_z_rot)
    else:
        z_angle = 0

    x_radians = np.pi*(x_angle/180)
    x_rotmat = torch.FloatTensor(get_rotmat_x(x_radians))

    y_radians = np.pi*(y_angle/180)
    y_rotmat = torch.FloatTensor(get_rotmat_y(y_radians))

    z_radians = np.pi*(z_angle/180)
    z_rotmat = torch.FloatTensor(get_rotmat_z(z_radians))
    aug_rotmat = torch.matmul(torch.matmul(z_rotmat, y_rotmat), x_rotmat)
    return aug_rotmat

def get_augm_trans(x_trans, y_trans, z_trans):
    aug_x_trans = 2*x_trans*torch.rand(1)-x_trans # [-x_trans, x_trans]
    aug_y_trans = 2*y_trans*torch.rand(1)-y_trans # [-y_trans, y_trans]
    aug_z_trans = 2*z_trans*torch.rand(1)-z_trans # [-z_trans, z_trans]
    aug_trans = torch.stack([aug_x_trans, aug_y_trans, aug_z_trans])
    return aug_trans

def get_rotmat_x(radians):
    x_rotmat = np.array(
        [
            [1,                 0,                  0],
            [0,   np.cos(radians),   -np.sin(radians)],
            [0,   np.sin(radians),    np.cos(radians)]
        ]
    )
    return x_rotmat

def get_rotmat_y(radians):
    y_rotmat = np.array(
        [
            [ np.cos(radians),  0, np.sin(radians)],
            [               0,  1,               0],
            [-np.sin(radians),  0, np.cos(radians)]
        ]
    )
    return y_rotmat

def get_rotmat_z(radians):
    z_rotmat = np.array(
        [
            [np.cos(radians), -np.sin(radians), 0], 
            [np.sin(radians),  np.cos(radians), 0], 
            [                0,                  0, 1], 
        ]
    )
    return z_rotmat

