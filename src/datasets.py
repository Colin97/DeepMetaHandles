import torch
import numpy as np
import os
import random

def my_collate(batch):
    src_pc = []
    src_name = []
    key_pts = []
    tar_pc = []
    tar_name = []
    w_pc = []
    src_ver = []
    src_face = []
    tar_ver = []
    tar_face = []
    real_ver = []
    real_face = []
    w_mesh = []
    for data in batch:
        src_pc.append(torch.from_numpy(data["src_pc"]).unsqueeze(0).float())
        src_name.append(data["src_name"])
        tar_pc.append(torch.from_numpy(data["tar_pc"]).unsqueeze(0).float())
        tar_name.append(data["tar_name"])
        key_pts.append(torch.from_numpy(data["key_pts"]).unsqueeze(0).float())
        w_pc.append(torch.from_numpy(data["w_pc"]).unsqueeze(0).float())
        src_ver.append(torch.from_numpy(data["src_ver"]))
        src_face.append(torch.from_numpy(data["src_face"]))
        tar_ver.append(torch.from_numpy(data["tar_ver"]))
        tar_face.append(torch.from_numpy(data["tar_face"]))
        real_ver.append(torch.from_numpy(data["real_ver"]))
        real_face.append(torch.from_numpy(data["real_face"]))
        w_mesh.append(torch.from_numpy(data["w_mesh"]).float())
    src_pc = torch.cat(src_pc).cuda()
    tar_pc = torch.cat(tar_pc).cuda()
    key_pts = torch.cat(key_pts).cuda()
    w_pc = torch.cat(w_pc).cuda()
    return {"src_pc": src_pc, "src_name": src_name, "key_pts": key_pts,
            "tar_pc": tar_pc, "tar_name": tar_name, "w_pc": w_pc,
            "src_ver": src_ver, "src_face": src_face, "w_mesh": w_mesh,
            "tar_ver": tar_ver, "tar_face": tar_face,
            "real_ver": real_ver, "real_face": real_face}


class ChairDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir="../data/chair"):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase

        with open(os.path.join(self.data_dir, "all.txt")) as f:
            lines = f.readlines()
            self.models = [line.rstrip() for line in lines]

        self.k = 20

        with open(os.path.join(self.data_dir, "%s.txt" % self.phase)) as f:
            lines = f.readlines()
            self.ids = [int(line.rstrip()) for line in lines]

        self.pc = np.load(os.path.join(
            self.data_dir, "pc_4096.npy"), allow_pickle=True)
        self.key_pts = np.load(os.path.join(
            self.data_dir, "key_point_50.npy"), allow_pickle=True)
        self.mesh_vertices = np.load(os.path.join(
            self.data_dir, "mesh_vertices.npy"), allow_pickle=True)
        self.mesh_faces = np.load(os.path.join(
            self.data_dir, "mesh_faces.npy"), allow_pickle=True)
        # biharmonic weights
        self.w_pc = np.load(os.path.join(
            self.data_dir, "w_pc_4096.npy"), allow_pickle=True)
        w_mesh_0 = np.load(os.path.join(
            self.data_dir, "w_mesh_0.npy"), allow_pickle=True)
        w_mesh_1 = np.load(os.path.join(
            self.data_dir, "w_mesh_1.npy"), allow_pickle=True)
        w_mesh_2 = np.load(os.path.join(
            self.data_dir, "w_mesh_2.npy"), allow_pickle=True)
        w_mesh_3 = np.load(os.path.join(
            self.data_dir, "w_mesh_3.npy"), allow_pickle=True)
        self.w_mesh = list(w_mesh_0) + list(w_mesh_1) + \
            list(w_mesh_2) + list(w_mesh_3)

    def __len__(self):
        return len(self.ids) * self.k

    def __getitem__(self, idx):
        src_id = self.ids[idx // self.k]
        tar_id = random.choice(self.ids)
        # positive sample for discriminator network
        real_id = random.choice(self.ids) 
        src_name = self.models[src_id]
        tar_name = self.models[tar_id]
        src_pc = self.pc[src_id]
        tar_pc = self.pc[tar_id]
        key_pts = self.key_pts[src_id]
        w_mesh = self.w_mesh[src_id]
        w_pc = self.w_pc[src_id]
        src_ver = self.mesh_vertices[src_id]
        src_face = self.mesh_faces[src_id]
        tar_ver = self.mesh_vertices[tar_id]
        tar_face = self.mesh_faces[tar_id]
        real_ver = self.mesh_vertices[real_id]
        real_face = self.mesh_faces[real_id]
        return {"src_name": src_name, "tar_name": tar_name,
                "src_pc": src_pc, "tar_pc": tar_pc,
                "src_ver": src_ver, "src_face": src_face,
                "tar_ver": tar_ver, "tar_face": tar_face,
                "real_ver": real_ver, "real_face": real_face,
                "key_pts": key_pts, "w_mesh": w_mesh, "w_pc": w_pc}
