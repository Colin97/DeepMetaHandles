import os
import numpy as np
from pytorch3d.io import load_obj

categories = ['04379243', '04256520']
root = '/YOUR/DATA/DIR'

for cat in categories:
    with open("all_%s.txt" % cat) as f:
        lines = f.readlines()
        
    key_point_50_list = []
    for line in lines:
        model = line.rstrip()
        vertices, faces, _ = load_obj("%s/key_point_50/%s/%s.obj"%(root, cat, model))
        key_point_50_list.append(vertices.numpy())

    np.save("%s/key_point_50_%s.npy"%(root, cat), key_point_50_list)