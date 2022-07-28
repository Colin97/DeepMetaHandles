import os
import numpy as np


categories = ['04379243', '04256520']
root = '/YOUR/DATA/DIR'

def load_txt(file_name):
    with open(file_name) as f:
        data = f.readlines()
    n = int(data[0].split(" ")[0])
    m = int(data[0].split(" ")[1])
    weights = np.zeros((n, m))
    for i in range(n):
        w = data[i + 1].split(" ")
        for j in range(m):
            weights[i][j] = float(w[j])
    return weights

for cat in categories:
    with open("all_%s.txt" % cat) as f:
        lines = f.readlines()
    
    w_mesh_list = []
    w_pc_4096_list = []
    for line in lines:
        model = line.rstrip()

        w_mesh = load_txt("%s/w_mesh/%s/%s.txt"%(root, cat, model))
        #np.save("%s/w_mesh/%s/%s.npy"%(root, cat, model), w_mesh)

        if w_mesh.min() < -1.5 or w_mesh.max() > 1.5:
            print("[Warning!] Weird weights detected! Please ignore %s during training." % model)
        
        w_pc_4096 = load_txt("%s/w_pc_4096/%s/%s.txt"%(root, cat, model))
        #np.save("%s/w_pc_4096/%s/%s.npy"%(root, cat, model), w_mesh)

        w_mesh_list.append(w_mesh)
        w_pc_4096_list.append(w_pc_4096)

    np.save("/%s/w_mesh_%s.npy"%(root, cat), w_mesh_list, allow_pickle = True)
    np.save("/%s/w_pc_4096_%s.npy"%(root, cat), w_pc_4096_list, allow_pickle = True)