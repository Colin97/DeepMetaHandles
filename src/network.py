import torch
import torch.nn.functional as F
import torch.nn as nn
from pointnet_utils import pointnet_encoder
from losses import chamfer_distance

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(32, 32, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc = nn.Linear(8192, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.leakyrelu1(self.bn1(self.conv1(x)))
        x = self.leakyrelu2(self.bn2(self.conv2(x)))
        x = self.leakyrelu3(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.sigmoid(self.fc(x))
        return x

class model(nn.Module):
    def __init__(self, num_basis):
        super(model, self).__init__()
        self.pointnet = pointnet_encoder()
        self.num_basis = num_basis

        # src point feature 2883 * N
        self.conv11 = torch.nn.Conv1d(2883, 128, 1)
        self.conv12 = torch.nn.Conv1d(128, 64, 1)
        self.conv13 = torch.nn.Conv1d(64, 64, 1)
        self.bn11 = nn.BatchNorm1d(128)
        self.bn12 = nn.BatchNorm1d(64)
        self.bn13 = nn.BatchNorm1d(64)

        # key point feature K (64 + 3 + 1) * N
        self.conv21 = torch.nn.Conv1d(68, 64, 1)
        self.conv22 = torch.nn.Conv1d(64, 64, 1)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)

        # basis feature K 64
        self.conv31 = torch.nn.Conv1d(64 + 3, 256, 1)
        self.conv32 = torch.nn.Conv1d(256, 512, 1)
        self.conv33 = torch.nn.Conv1d(512, self.num_basis * 3, 1)
        self.bn31 = nn.BatchNorm1d(256)
        self.bn32 = nn.BatchNorm1d(512)

        # key point feature with target K (2048 + 64 + 3)
        self.conv41 = torch.nn.Conv1d(2048 + 64 + 3, 256, 1)
        self.conv42 = torch.nn.Conv1d(256, 128, 1)
        self.conv43 = torch.nn.Conv1d(128, 128, 1)
        self.bn41 = nn.BatchNorm1d(256)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn43 = nn.BatchNorm1d(128)

        # coef feature 15 (128 + 3 + 3) K
        self.conv51 = torch.nn.Conv1d(128 + 3 + 3, 256, 1)
        self.conv52 = torch.nn.Conv1d(256, 128, 1)
        self.conv53 = torch.nn.Conv1d(128, 128, 1)
        self.bn51 = nn.BatchNorm1d(256)
        self.bn52 = nn.BatchNorm1d(128)
        self.bn53 = nn.BatchNorm1d(128)

        # coef feature 15 128
        self.conv61 = torch.nn.Conv1d(128 + 2, 64, 1)
        self.conv62 = torch.nn.Conv1d(64, 32, 1)
        self.conv63 = torch.nn.Conv1d(32, 1, 1)
        self.bn61 = nn.BatchNorm1d(64)
        self.bn62 = nn.BatchNorm1d(32)

        self.conv71 = torch.nn.Conv1d(64 + 3 + 3, 32, 1)
        self.conv72 = torch.nn.Conv1d(32, 16, 1)
        self.conv73 = torch.nn.Conv1d(16, 2, 1)
        self.bn71 = nn.BatchNorm1d(32)
        self.bn72 = nn.BatchNorm1d(16)

        self.sigmoid = nn.Sigmoid()

    def forward(self, src_pc, tar_pc, key_pts, w_pc):
        B, N, _ = src_pc.shape
        src_out, src_global = self.pointnet(src_pc, False)
        tar_global = self.pointnet(tar_pc, True)

        src_out = F.relu(self.bn11(self.conv11(src_out)))
        src_out = F.relu(self.bn12(self.conv12(src_out)))
        src_out = F.relu(self.bn13(self.conv13(src_out)))

        _, K, _ = key_pts.shape
        key_pts1 = key_pts.unsqueeze(-1).expand(-1, -1, -1, N)  # B K 3 N
        w_pc1 = w_pc.transpose(2, 1).unsqueeze(2)  # B K 1 N
        src_out = src_out.unsqueeze(1).expand(-1, K, -1, -1)  # B K 64 N
        net = torch.cat([src_out, w_pc1, key_pts1], 2).view(B * K, 68, N)

        net = F.relu(self.bn21(self.conv21(net)))
        net = self.bn22(self.conv22(net))

        net = torch.max(net, 2, keepdim=True)[0]
        key_fea = net.view(B * K, 64, 1)

        net = torch.cat([key_fea, key_pts.view(B * K, 3, 1)], 1)
        net = F.relu(self.bn31(self.conv31(net)))
        net = F.relu(self.bn32(self.conv32(net)))
        basis = self.conv33(net).view(B, K * 3, self.num_basis).transpose(1, 2)
        basis = basis / basis.norm(p=2, dim=-1, keepdim=True) 

        key_fea_range = key_fea.view(
            B, K, 64, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
        key_pts_range = key_pts.view(
            B, K, 3, 1).expand(-1, -1, -1, self.num_basis).transpose(1, 3)
        basis_range = basis.view(B, self.num_basis, K, 3).transpose(2, 3)

        coef_range = torch.cat([key_fea_range, key_pts_range, basis_range], 2).view(
            B * self.num_basis, 70, K)
        coef_range = F.relu(self.bn71(self.conv71(coef_range)))
        coef_range = F.relu(self.bn72(self.conv72(coef_range)))
        coef_range = self.conv73(coef_range)
        coef_range = torch.max(coef_range, 2, keepdim=True)[0]
        coef_range = coef_range.view(B, self.num_basis, 2) * 0.01
        coef_range[:, :, 0] = coef_range[:, :, 0] * -1

        src_tar = torch.cat([src_global, tar_global], 1).unsqueeze(
            1).expand(-1, K, -1).reshape(B * K, 2048, 1)

        key_fea = torch.cat([key_fea, src_tar, key_pts.view(B * K, 3, 1)], 1)
        key_fea = F.relu(self.bn41(self.conv41(key_fea)))
        key_fea = F.relu(self.bn42(self.conv42(key_fea)))
        key_fea = F.relu(self.bn43(self.conv43(key_fea)))

        key_fea = key_fea.view(B, K, 128).transpose(
            1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
        key_pts2 = key_pts.view(B, K, 3).transpose(
            1, 2).unsqueeze(1).expand(-1, self.num_basis, -1, -1)
        basis1 = basis.view(B, self.num_basis, K, 3).transpose(2, 3)

        net = torch.cat([key_fea, basis1, key_pts2], 2).view(
            B * self.num_basis, 3 + 128 + 3, K)

        net = F.relu(self.bn51(self.conv51(net)))
        net = F.relu(self.bn52(self.conv52(net)))
        net = self.bn53(self.conv53(net))

        net = torch.max(net, 2, keepdim=True)[0]
        net = net.view(B * self.num_basis, 128, 1)

        net = torch.cat([net, coef_range.view(B * self.num_basis, 2, 1)], 1)
        net = F.relu(self.bn61(self.conv61(net)))
        net = F.relu(self.bn62(self.conv62(net)))
        coef = self.sigmoid(self.conv63(net)).view(B, self.num_basis)

        coef = (coef * coef_range[:, :, 0] + (1 - coef)
                * coef_range[:, :, 1]).view(B, 1, self.num_basis)

        def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3)
        def_pc = torch.bmm(w_pc, def_key_pts)

        cd_loss = chamfer_distance(def_pc, tar_pc)

        ratio = torch.rand((B, self.num_basis)).cuda()
        sample_coef = (ratio * coef_range[:, :, 0] + (1 - ratio)
                       * coef_range[:, :, 1]).view(B, 1, self.num_basis)
        sample_def_key_pts = key_pts + \
            torch.bmm(sample_coef, basis).view(B, K, 3)
        sample_def_pc = torch.bmm(w_pc, sample_def_key_pts)
        sample_def_pc_sym = sample_def_pc * \
            torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
        sym_loss = chamfer_distance(sample_def_pc, sample_def_pc_sym)
        return def_key_pts, def_pc, cd_loss, basis, coef, sample_def_key_pts, sym_loss, coef_range
