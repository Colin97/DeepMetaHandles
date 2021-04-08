import torch
import numpy as np
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    Textures,
    look_at_view_transform,
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    RasterizationSettings
)

def render_mesh(verts, faces):
    device = verts[0].get_device()
    N = len(verts)
    num_verts_per_mesh = []
    for i in range(N):
        num_verts_per_mesh.append(verts[i].shape[0])
    verts_rgb = torch.ones((N, np.max(num_verts_per_mesh), 3),
                            requires_grad=False, device=device)
    for i in range(N):
        verts_rgb[i, num_verts_per_mesh[i]:, :] = -1
    textures = Textures(verts_rgb=verts_rgb)

    meshes = Meshes(verts=verts,
                    faces=faces,
                    textures=textures)
    elev = torch.rand(N) * 30 - 15
    azim = torch.rand(N) * 360 - 180
    R, T = look_at_view_transform(dist=2, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    sigma = 1e-4
    raster_settings = RasterizationSettings(
        image_size=128,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=40,
        perspective_correct=False
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader()
    )
    return renderer(meshes)
