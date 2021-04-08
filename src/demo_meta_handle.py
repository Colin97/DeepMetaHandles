import torch
import numpy as np
import network
import argparse
import os
from pytorch3d.io import load_obj, save_obj

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--src_shape', type=str, default="../data/demo/104256", help='source shape')
    parser.add_argument('--tar_shape', type=str, default="../data/demo/834af6", help='target shape')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/chair_15.pth", help='checkpoint path')
    parser.add_argument('--save_dir', type=str, default="104256_meta_handle", help='save dir')
    parser.add_argument('--num_basis', type=int, default=15, help='number of basis vectors')
    return parser.parse_args()
    
opt = parse_args()
net = network.model(opt.num_basis).cuda()
net = torch.nn.DataParallel(net)
try:
    checkpoint = torch.load(opt.checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrain model')
except:
    print('Error! No existing model!')
    exit(-1)
    
net.eval()

src_pc, _, _ = load_obj("%s/pc_4096.obj" % opt.src_shape)
tar_pc, _, _ = load_obj("%s/pc_4096.obj" % opt.tar_shape)
key_pts, _, _ = load_obj("%s/key_point_50.obj" % opt.src_shape)
_, src_faces, _ = load_obj("%s/manifold.obj" % opt.src_shape)
src_faces = src_faces.verts_idx
w_pc = torch.from_numpy(np.load("%s/w_pc_4096.npy" % opt.src_shape))
w_mesh = torch.from_numpy(np.load("%s/w_mesh.npy" % opt.src_shape))
src_pc = src_pc.unsqueeze(0).cuda()
tar_pc = tar_pc.unsqueeze(0).cuda()
key_pts = key_pts.unsqueeze(0).cuda()
w_pc = w_pc.unsqueeze(0).cuda()

_, _, _, basis, _, _, _, coef_range = net(src_pc, tar_pc, key_pts, w_pc)

os.makedirs(opt.save_dir, exist_ok=True)
for i in range(opt.num_basis):
    l = coef_range[0][i][0] * 2
    r = coef_range[0][i][1] * 2
    for j in range(4):
        scale = (r - l) / 3 * j + l
        off = basis[0][i].reshape(50, 3) * scale
        def_key_pts = key_pts[0] + off
        def_ver = torch.matmul(w_mesh.cuda(), def_key_pts)
        save_obj(("%s/%d-%d.obj") % (opt.save_dir, i, j), def_ver.cpu(), src_faces)
            