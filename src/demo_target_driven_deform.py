import torch
import numpy as np
import network
import argparse
import os
from pytorch3d.io import load_obj, save_obj

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--src_shape', type=str, default="../data/demo/2e0beb", help='source shape')
    parser.add_argument('--tar_shape', type=str, default="../data/demo/834af6", help='target shape')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/chair_15.pth", help='checkpoint path') 
    parser.add_argument('--save_name', type=str, default="2e0beb-834af6-def.obj", help='save name')
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

def_key_pts, _, _, _, _, _, _, _ = net(src_pc, tar_pc, key_pts, w_pc)

def_ver = torch.matmul(w_mesh.cuda(), def_key_pts[0])
save_obj(opt.save_name, def_ver.cpu(), src_faces)