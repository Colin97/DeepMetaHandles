import os
import numpy as np
from pytorch3d.io import load_obj

categories = ['04379243', '04256520']
root = '/YOUR/DATA/DIR'

for cat in categories:
    with open("all_%s.txt" % cat) as f:
        lines = f.readlines()
        
    mesh_vertices_list = []
    mesh_faces_list = []
    for line in lines:
        model = line.rstrip()
        vertices, faces, _ = load_obj("%s/tet/%s/%s.mesh__sf.obj"%(root, cat, model))
        mesh_vertices_list.append(vertices.numpy())
        mesh_faces_list.append(faces.verts_idx.numpy())

    np.save("%s/mesh_vertices_%s.npy"%(root, cat), mesh_vertices_list)
    np.save("%s/mesh_faces_%s.npy"%(root, cat), mesh_faces_list)