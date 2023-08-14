import numpy as np
import torch
import trimesh
import math
import torch.nn.functional as F
from src.utils.mesh_utils import MeshViewer
from src.utils.smpl_body_utils import colors,marker2bodypart,bodypart2color
from src.utils.smpl_body_utils import c2rgba, get_body_model
from hydra.utils import get_original_cwd
from src.render.video import save_video_samples

import os 
from PIL import Image
import subprocess
import contextlib

def visualize_meshes(vertices, pcd=None, multi_col=None, text=None,
                     multi_angle=False, h=720, w=720, bg_color='white',
                     save_path=None, fig_label=None, use_hydra_path=True):
    import pyrender

    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )
 
    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height: int = h
    im_width: int = w
    vis_mar= True if pcd is not None else False

    # if multi_angle:
    #     view_angles = [0, 180, 90, -90]
    # else:
    #     view_angles = [0]
    seqlen = vertices.shape[0]
    kfs_viz = np.zeros((seqlen))
    if multi_col is not None:
        # assert len(set(multi_col)) == len(multi_col)
        multi_col = multi_col.detach().cpu().numpy()
        for i, kf_ids in enumerate(multi_col):
            kfs_viz[i, :].put(np.round_(kf_ids).astype(int), 1, mode='clip')
    

    if isinstance(pcd, np.ndarray):
        pcd = torch.from_numpy(pcd)

    if vis_mar:
        pcd = pcd.reshape(seqlen, 67, 3).to('cpu')
        if len(pcd.shape) == 3:
            pcd = pcd.unsqueeze(0)
    mesh_rec = vertices
    if use_hydra_path:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            smpl = get_body_model(path=f'{get_original_cwd()}/data/smpl_models',
                                model_type='smpl', gender='neutral',
                                batch_size=1, device='cpu')
    else:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            smpl = get_body_model(path=f'data/smpl_models',
                                model_type='smpl', gender='neutral',
                                batch_size=1, device='cpu')
    minx, miny, _ = mesh_rec.min(axis=(0, 1))
    maxx, maxy, _ = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 2])  # Min height

    mesh_rec = mesh_rec.copy()
    mesh_rec[:, :, 2] -= height_offset

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy, 
                    use_offscreen=True,
                    bg_color=bg_color)
                #ground_height=(mesh_rec.detach().cpu().numpy()[0, 0, 6633, 2] - 0.01))
    # ground plane has a bug if we want batch_size to work
    mv.render_wireframe = False
    
    video = np.zeros([seqlen, im_height, im_width, 3])
    for i in range(seqlen):
        Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])

        if vis_mar:
            m_pcd = trimesh.points.PointCloud(pcd[i])

        if kfs_viz[i]:
            mesh_color = np.tile(c2rgba(colors['light_grey']), (6890, 1))
        else:
            mesh_color = np.tile(c2rgba(colors['yellow_pale']), (6890, 1))
        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=smpl.faces,
                                vertex_colors=mesh_color)

        m_rec.apply_transform(Rx)
        m_rec.apply_transform(Ry)

        all_meshes = []
        if vis_mar:
            m_pcd.apply_transform(Rx)
            m_pcd.apply_transform(Ry)

            m_pcd = np.array(m_pcd.vertices)
            # after trnaofrming poincloud visualize for each body part separately
            pcd_bodyparts = dict()
            for bp, ids in marker2bodypart.items():
                points = m_pcd[ids]
                tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
                tfs[:, :3, 3] = points
                col_sp = trimesh.creation.uv_sphere(radius=0.01)

                # debug markers, maybe delete it
                # if bp == 'special':
                #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

                if kfs_viz[i]:
                    col_sp.visual.vertex_colors = c2rgba(colors["black"])
                else:
                    col_sp.visual.vertex_colors = c2rgba(colors[bodypart2color[bp]])

                pcd_bodyparts[bp] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)

            for bp, m_bp in pcd_bodyparts.items():
                all_meshes.append(m_bp)


        all_meshes.append(m_rec)
        mv.set_meshes(all_meshes, group_name='static')
        video[i] = mv.render()

    # if save_path is not None:
    #     save_video_samples(np.transpose(np.squeeze(video),
    #                                     (0, 3, 1, 2)).astype(np.uint8), 
    #                                     save_path, write_gif=True)
    #     return
    if multi_angle:
        return np.transpose(video, (0, 1, 4, 2, 3)).astype(np.uint8)

    if save_path is not None:
        from src.render.video import Video
        vid = Video(list(np.squeeze(video).astype(np.uint8)))
        if text is not None:
            vid.add_text(text)
        return vid.save(save_path)
        # return save_video_samples(np.transpose(np.squeeze(video),
        #                                        (0, 3, 1, 2)).astype(np.uint8),
        #                           save_path, text)
    return np.transpose(np.squeeze(video), (0, 3, 1, 2)).astype(np.uint8)
