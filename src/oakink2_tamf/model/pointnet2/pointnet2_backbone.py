import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg
import torch


class PointNet2_backbone(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNet2_backbone, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        local_features = l1_points.permute(0, 2, 1).contiguous()  # B, 512, 128
        global_features = l3_points.view(B, 1024)
        local_xyz = l1_xyz.permute(0, 2, 1).contiguous()  # B, 512, 3

        return global_features, local_features, local_xyz
    
class PointNet2_backbone_msg(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNet2_backbone_msg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc = nn.Linear(320, 128)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        local_features = l1_points.permute(0, 2, 1).contiguous()  # B, 512, 320
        local_features = self.fc(local_features)
        global_features = l3_points.view(B, 1024)

        return global_features, local_features
    
# def main():
#     pointnet2_backbone = PointNet2_backbone(normal_channel=True)
#     input_tensor = torch.randn(2, 6, 8192)
#     global_features, local_features, l1_xyz = pointnet2_backbone(input_tensor)
#     print("Global Features Shape:", global_features.shape)  # Expected: (2, 1024)
#     print("Local Features Shape:", local_features.shape)    # Expected: (2, 512, 128)
#     print("L1 XYZ Shape:", l1_xyz.shape)                  # Expected: (2, 512, 3)

# if __name__ == "__main__":
#     main()
