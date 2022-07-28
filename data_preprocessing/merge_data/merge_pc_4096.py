import os
import numpy as np
from pytorch3d.io import load_obj

categories = ['04379243', '04256520']
root = '/YOUR/DATA/DIR'

for cat in categories:
    with open("all_%s.txt" % cat) as f:
        lines = f.readlines()

    pc_4096_list = []
    for line in lines:
        model = line.rstrip()
        vertices, faces, _ = load_obj("%s/pc_4096/%s/%s.obj"%(root, cat, model))
        pc_4096_list.append(vertices.numpy())
        
    np.save("%s/pc_4096_%s.npy"%(root, cat), pc_4096_list)