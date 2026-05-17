import os
import numpy as np
import torch
import logging

from manotorch.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from oakink2_tamf.model.loss.chamfer_distance import point2point_signed
from oakink2_tamf.model.arctic_mano.mano import build_mano_aa
from oakink2_tamf.model.arctic_mano.rot import rot6d_to_axis_angle, rot6d_to_rotmat, axis_angle_to_quaternion, quaternion_apply

_logger = logging.getLogger(__name__)

class GenModelLoss(torch.nn.Module): 
    def __init__(self, mano_path, loss_cfg, use_pc=True):
        super().__init__()

        self.mano_lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
        self.mano_rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False)


        self.register_buffer("vpe", torch.from_numpy(np.load(loss_cfg["vpe_path"])).to(torch.long))
        self.register_buffer("v_weights", torch.from_numpy(np.load(loss_cfg["c_weight_path"])).to(torch.float32))
        self.register_buffer("v_weights2", torch.pow(self.v_weights, 1.0 / 2.5))
        self.register_buffer("contact_v", self.v_weights > 0.8)
        self.use_pc = use_pc

        self.loss_type = loss_cfg["loss_type"]
        _logger.info("loss type: %s", self.loss_type)
        self.enable_dist_h_loss = False
        self.enable_rec_joint_loss = False
        self.enable_rec_vert_loss = False
        self.enable_rec_mano_loss = False
        self.enable_dist_o_loss = False
        self.enable_edge_len_loss = False
        self.enable_penetration_loss = False
        self.enable_contact_loss = False
        self.enable_contact_region_loss = False
        self.enable_smooth_loss = False
        # baseline
        if 'h' in self.loss_type:
            self.enable_dist_h_loss = True
        if 'j' in self.loss_type:
            self.enable_rec_joint_loss = True
        if 'v' in self.loss_type:
            self.enable_rec_vert_loss = True
        # add
        if 'm' in self.loss_type:
            self.enable_rec_mano_loss = True
        if 'o' in self.loss_type:
            self.enable_dist_o_loss = True
        if 'e' in self.loss_type:
            self.enable_edge_len_loss = True
        if 'p' in self.loss_type:
            self.enable_penetration_loss = True
        if 'c' in self.loss_type:
            self.enable_contact_loss = True
        if 'r' in self.loss_type:
            self.enable_contact_region_loss = True
        if 's' in self.loss_type:
            self.enable_smooth_loss = True


    def proc_obj(self, obj_verts, obj_params, obj_top_idx=None):
        nframes = obj_params.shape[0]
        obj_trans = obj_params[:, :3]
        obj_rot6d = obj_params[:, 3:9]
        obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
        if obj_params.shape[-1] == 10 and obj_top_idx is not None:
            obj_top_idx = obj_top_idx.bool()
            obj_angle = obj_params[..., 9:10]
            quat_arti = axis_angle_to_quaternion(torch.FloatTensor([0, 0, -1]).to(obj_params.device).view(1, 3)*obj_angle) # seq, 4
            obj_verts = obj_verts.unsqueeze(0).expand(nframes, -1, -1)
            obj_verts2 = obj_verts.clone()
            obj_top_idx = obj_top_idx.unsqueeze(0).expand(nframes, -1)
            obj_verts2[obj_top_idx] = quaternion_apply(quat_arti[:, None], obj_verts)[obj_top_idx]
            obj_pc_rotated = torch.einsum("tij,tkj->tki", obj_rotmat, obj_verts2)
        obj_verts_transformed = obj_pc_rotated+obj_trans.unsqueeze(1)
        return obj_verts_transformed  # (nframes, 1024, 3)
    

    def proc_hand(self, x_hand, hand_shape, hand_side):

        # hand
        hand_pose = x_hand[:, 3:]
        hand_pose = rot6d_to_axis_angle(hand_pose).reshape(-1, 48)
        hand_trans = x_hand[:, :3]
        if hand_side == "lh":
            out = self.mano_lhand_layer(
                global_orient=hand_pose[:, :3],
                hand_pose=hand_pose[:, 3:48],
                betas=hand_shape.to(hand_pose.device)
            )
            hand_faces = self.mano_lhand_layer.faces.copy().astype(np.int16)
        elif hand_side == "rh":
            out = self.mano_rhand_layer(
                global_orient=hand_pose[:, :3],
                hand_pose=hand_pose[:, 3:48],
                betas=hand_shape.to(hand_pose.device)
            )
            hand_faces = self.mano_rhand_layer.faces.copy().astype(np.int16)

        hand_trans = hand_trans.unsqueeze(1)
        hand_verts = out.vertices + hand_trans
        hand_joints = out.joints_w_tip + hand_trans
        hand_faces = torch.LongTensor(hand_faces).to(hand_pose.device)
            
        return hand_verts, hand_joints, hand_faces  # (seq, 778, 3), (seq, 21, 3), (1538, 3)
    

    def _edges_for(self, x, vpe):
        return x[:, vpe[:, 0]] - x[:, vpe[:, 1]]
    
    def sum_flat(self, tensor):
        """
        Take the sum over all non-batch dimensions.
        """
        return tensor.sum(dim=list(range(1, len(tensor.shape))))
    
    def masked_l2(self, a, b, mask):
        l2_loss = (
            lambda a, b: (a - b) ** 2
        )  
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = l2_loss(a, b)
        loss = self.sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = self.sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / (non_zero_elements + 1e-6)
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    def forward(self, model_output, batch):
        
        # model_output: (bs, in_dim, 1, seq)
        batch_size = model_output.shape[0]
        seq_len = model_output.shape[3]

        loss_rec_joint = 0.0
        loss_rec_vert = 0.0
        loss_edge_len = 0.0
        loss_penetration = 0.0
        loss_contact = 0.0
        loss_contact_region = 0.0
        loss_smooth = 0.0
        loss_dist_h = 0.0
        loss_dist_o = 0.0
        param_loss = 0.0

        batch_avai_len = batch["len"]

        # parameters loss
        if self.enable_rec_mano_loss:
            gt = batch["pose_repr"].unsqueeze(3)  # (bs, seqlen, in_dim, 1)
            gt = gt.permute((0, 2, 3, 1))  # (bs, in_dim, 1, seqlen)
            assert model_output.shape == gt.shape, f"{model_output.shape}" + f" != {gt.shape}"
            param_loss = self.masked_l2(model_output, gt, batch["mask"].reshape((batch_size, 1, 1, seq_len)))
            param_loss = param_loss.mean()


        for batch_offset in range(batch_size):
            # decode pose from model and from gt
            hand_side = batch["hand_side"][batch_offset]
            shape = batch["shape"][batch_offset]
            obj_list = batch["obj_list"][batch_offset]
            if self.use_pc:
                obj_verts_list = batch["obj_pointcloud"][batch_offset, :, :, 0:3]
            else:
                obj_verts_list = batch["obj_verts"][batch_offset]
            obj_pc_top_idx = batch["obj_pc_top_idx"][batch_offset]
            assert obj_verts_list.shape[-1] == 3

            obj_traj = batch["obj_traj"][batch_offset]
            mask = batch["mask"][batch_offset]  # [seqlen, ]
            with torch.no_grad():
                mask_coef = float(mask.shape[0] / (torch.sum(mask) + 1e-6))

            pose_repr_gt = batch["pose_repr"][batch_offset]
            pose_repr_pred = model_output[batch_offset].permute((2, 0, 1)).squeeze(-1)  # [seqlen, 99]
            verts_gt, joints_gt, faces_gt = self.proc_hand(pose_repr_gt, shape, hand_side)  # [seqlen, n, 3]
            verts_pred, joints_pred, faces_pred = self.proc_hand(pose_repr_pred, shape, hand_side)  # [seqlen, n, 3]
            # verts_gt, joints_gt = self.recover_mano_from_pose_repr(pose_repr_gt, shape, hand_side)  # [seqlen, n, 3]
            # verts_pred, joints_pred = self.recover_mano_from_pose_repr(pose_repr_pred, shape, hand_side)
            mesh_gt = Meshes(verts=verts_gt, faces=faces_gt.unsqueeze(0))
            mesh_pred = Meshes(verts=verts_pred, faces=faces_pred.unsqueeze(0))
            normals_gt = mesh_gt.verts_normals_packed().view((-1, 778, 3))
            normals_pred = mesh_pred.verts_normals_packed().view((-1, 778, 3))

            # joint loss
            if self.enable_rec_joint_loss:
                joint_dist_sq = torch.sum(torch.pow(joints_pred - joints_gt, exponent=2), dim=-1)  # [seqlen, n]  distance^2
                joint_dist_sq = joint_dist_sq * mask.unsqueeze(1)
                joint_loss = mask_coef * torch.mean(joint_dist_sq)
                loss_rec_joint = loss_rec_joint + joint_loss

            # verts loss
            if self.enable_rec_vert_loss:
                vert_dist_sq = torch.sum(torch.pow(verts_pred - verts_gt, exponent=2), dim=-1)  # [seqlen, n]
                vert_dist_sq = vert_dist_sq * mask.unsqueeze(1)
                vert_loss = mask_coef * torch.mean(
                    torch.einsum(
                        "ij,j->ij",
                        vert_dist_sq,
                        torch.pow(self.v_weights, 2),
                    )
                )  # [1]
                loss_rec_vert = loss_rec_vert + vert_loss

            # edge len loss
            if self.enable_edge_len_loss:
                edge_len_pred = self._edges_for(verts_pred, self.vpe)  # [seqlen, ne, 3]
                edge_len_gt = self._edges_for(verts_gt, self.vpe)  # [seqlen, ne, 3]
                edge_diff = torch.abs(edge_len_pred - edge_len_gt)
                edge_loss = mask_coef * torch.mean(edge_diff * mask.unsqueeze(1).unsqueeze(2))
                loss_edge_len = loss_edge_len + edge_loss


            if self.enable_dist_h_loss or self.enable_dist_o_loss or self.enable_penetration_loss or self.enable_contact_loss or self.enable_contact_region_loss:
                num_obj = len(obj_list)
                for obj_offset, obj_id in enumerate(obj_list):

                    obj_verts_can_sel = obj_verts_list[obj_offset].to(verts_gt)  # [1024, 3]
                    obj_pc_top_idx_sel = obj_pc_top_idx[obj_offset].to(verts_gt)  # [1024, ]
                    obj_traj_sel = obj_traj[obj_offset]  # [seq_len, 10]
                    obj_verts_sel = self.proc_obj(obj_verts_can_sel, obj_traj_sel, obj_pc_top_idx_sel)  # [seq_len, 1024, 3]

                    n_obj_verts = obj_verts_sel.shape[1]
                    
                    o2h_signed, h2o, o2h_idx, h2o_idx = point2point_signed(verts_pred, obj_verts_sel, normals_pred)  # [seq_len, n_obj_verts]
                    o2h_signed_gt, h2o_gt, o2h_idx_gt, h2o_idx_gt = point2point_signed(verts_gt, obj_verts_sel, normals_gt)
                    verts_num = verts_gt.shape[1]

                    # h2o o2h loss
                    w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
                    w_dist_neg = o2h_signed < 0.0
                    w = torch.ones([seq_len, n_obj_verts]).to(verts_gt.device)
                    w[~w_dist] = 0.1  # less weight for far away vertices
                    w[w_dist_neg] = 1.5  # more weight for penetration

                    # dist_h
                    if self.enable_dist_h_loss:
                        dist_h = mask_coef * torch.mean(
                            torch.einsum("ij,j->ij", torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2)
                            * mask.unsqueeze(1)
                        )
                        loss_dist_h = loss_dist_h + (1.0 / num_obj) * dist_h

                    # dist_o
                    if self.enable_dist_o_loss:
                        dist_o = mask_coef * torch.mean(
                            torch.einsum("ij,ij->ij", torch.abs(o2h_signed - o2h_signed_gt), w) * mask.unsqueeze(1)
                        )
                        loss_dist_o = loss_dist_o + (1.0 / num_obj) * dist_o

                    # penetration_loss
                    if self.enable_penetration_loss:
                        # object
                        w_penetration = torch.zeros([seq_len, n_obj_verts]).to(verts_gt.device)
                        w_penetration[o2h_signed < 0.0] = 1  # penetration vertexs
                        mask_penetration = mask.unsqueeze(1).expand(-1, n_obj_verts) * w_penetration
                        with torch.no_grad():
                            penetration_coef = float(mask_penetration.shape[0] * mask_penetration.shape[1] / (torch.sum(mask_penetration) + 1e-6))
                        penetration_loss = torch.mean(torch.einsum("ij,ij->ij", torch.abs(o2h_signed), mask_penetration))
                        penetration_loss = penetration_coef * penetration_loss  # [1]
                        loss_penetration = loss_penetration + (1.0 / num_obj) * penetration_loss

                    # contact_loss
                    if self.enable_contact_loss:
                        # hand
                        w_gt = torch.ones([seq_len, verts_num]).to(verts_gt.device)
                        w_dist = (h2o_gt < 0.01)
                        w_gt[~w_dist] = 0  # no weight for far away vertices
                        mask_contact = mask.unsqueeze(1).expand(-1, verts_num) * w_gt
                        with torch.no_grad():
                            contact_coef = float(mask_contact.shape[0] * mask_contact.shape[1] / (torch.sum(mask_contact) + 1e-6))
                        h2o_weight = torch.einsum("ij,j->ij", h2o.abs(), self.v_weights2)
                        contact_loss = torch.mean(torch.einsum("ij,ij->ij", torch.abs(h2o_weight), mask_contact))
                        contact_loss = contact_coef * contact_loss  # [1]
                        loss_contact = loss_contact + (1.0 / num_obj) * contact_loss
                        # object
                        w_gt = torch.ones([seq_len, n_obj_verts]).to(verts_gt.device)
                        w_dist = (o2h_signed_gt < 0.01)
                        w_gt[~w_dist] = 0  # no weight for far away vertices
                        mask_contact = mask.unsqueeze(1).expand(-1, n_obj_verts) * w_gt
                        with torch.no_grad():
                            contact_coef = float(mask_contact.shape[0] * mask_contact.shape[1] / (torch.sum(mask_contact) + 1e-6))
                        contact_loss = torch.mean(torch.einsum("ij,ij->ij", torch.abs(o2h_signed), mask_contact))
                        contact_loss = contact_coef * contact_loss  # [1]
                        loss_contact = loss_contact + (1.0 / num_obj) * contact_loss

                    # obj contact_region_loss
                    if self.enable_contact_region_loss:
                        w_gt = torch.ones([seq_len, n_obj_verts]).to(verts_gt.device)
                        w_dist = (o2h_signed_gt < 0.005)
                        w_gt[~w_dist] = 0  # no weight for far away vertices
                        mask_contact_region = mask.unsqueeze(1).expand(-1, n_obj_verts) * w_gt
                        with torch.no_grad():
                            contact_region_coef = float(mask_contact_region.shape[0] * mask_contact_region.shape[1] / (torch.sum(mask_contact_region) + 1e-6))
                        contact_region_loss = torch.mean(torch.einsum("ij,ij->ij", torch.pow((o2h_idx/1000 - o2h_idx_gt/1000), exponent=2), mask_contact_region))
                        contact_region_loss = contact_region_coef * contact_region_loss  # [1]
                        loss_contact_region = loss_contact_region + (1.0 / num_obj) * contact_region_loss

            if self.enable_smooth_loss:   # 加速度
                v_pred = torch.diff(verts_pred, dim=0, n=2)
                avai_len = int(batch_avai_len[batch_offset])
                loss_smooth = loss_smooth + torch.mean(torch.abs(v_pred[:avai_len-2, ...]))

        loss_dist_h = loss_dist_h / batch_size
        loss_rec_joint = loss_rec_joint / batch_size
        loss_rec_vert = loss_rec_vert / batch_size
        loss_dist_o = loss_dist_o / batch_size
        loss_edge_len = loss_edge_len / batch_size
        loss_penetration = loss_penetration / batch_size
        loss_contact = loss_contact / batch_size
        loss_contact_region = loss_contact_region / batch_size
        loss_smooth = loss_smooth / batch_size

        loss = 0.0
        # baseline
        loss = loss + 0.1 * loss_dist_h
        loss = loss + 1.0 * loss_rec_joint
        loss = loss + 1.0 * loss_rec_vert
        loss = loss + 1.0 * param_loss
        loss = loss + 1.0 * loss_dist_o
        loss = loss + 0.1 * loss_edge_len
        # add
        loss = loss + 1.0 * loss_penetration
        loss = loss + 1.0 * loss_contact
        loss = loss + 0.1 * loss_contact_region   # change from 1.0 to 0.1
        loss = loss + 1.0 * loss_smooth
        loss_dict = {
            "loss": loss,
            "rec_param": param_loss,
            "rec_joint": loss_rec_joint,
            "rec_vert": loss_rec_vert,
            "edge_len": loss_edge_len,
            "dist_h": loss_dist_h,
            "dist_o": loss_dist_o,
            "penetration": loss_penetration,
            "contact": loss_contact,
            "contact_region": loss_contact_region,
            "smooth": loss_smooth,
        }

        loss = loss * 20  # 所有loss * 20

        return loss, loss_dict



if __name__ == "__main__":
    loss_cfg = {
        "vpe_path": "asset/grabnet/verts_per_edge.npy",
        "c_weight_path": "asset/grabnet/rhand_weight.npy",
        "loss_type": "h",
    }
    model_loss = GenModelLoss("asset/mano_v1_2", loss_cfg)
    x_hand, hand_shape, hand_side = torch.randn(100, 99), torch.randn(100, 10), "rh"
    x_obj = torch.randn(100, 10)
    obj_verts = torch.randn(1024, 3)
    obj_top_idx = torch.randint(0, 2, (1024,)).bool()

    hand_output = model_loss.proc_hand(x_hand, hand_shape, hand_side)
    obj_output = model_loss.proc_obj(obj_verts, x_obj, obj_top_idx)

    print("Hand Output:", hand_output[0].shape, hand_output[1].shape, hand_output[2].shape)
    print("Object Output:", obj_output.shape)
    