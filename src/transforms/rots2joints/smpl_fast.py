from __future__ import annotations

from typing import List, Optional, Dict, Callable

import sys
import os
import os.path as osp
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import smplx
from smplx.body_models import SMPLLayer, SMPLOutput
from smplx import lbs as lbs

from scipy.spatial.transform import Rotation as R

Tensor = torch.Tensor
def batch_size_from_tensor_list(tensor_list: List[Tensor]) -> int:
    batch_size = 1
    for tensor in tensor_list:
        if tensor is None:
            continue
        batch_size = max(batch_size, len(tensor))
    return batch_size


def identity_rot_mats(
    batch_size: int = 1,
    num_matrices: int = 1,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: Optional[torch.dtype] = torch.float32,
) -> Tensor:
    targs = {'dtype': dtype, 'device': device}
    return torch.eye(3, **targs).view(
        1, 1, 3, 3).repeat(batch_size, num_matrices, 1, 1)


class SMPLIndexed(SMPLLayer):
    def __init__(
        self,
        *args,
        vertex_indices=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert vertex_indices is not None
        vertex_indices = np.asarray(vertex_indices)
        self.num_vertices_to_keep = len(vertex_indices)

        joint_template = lbs.vertices2joints(
            self.J_regressor, self.v_template[None])
        self.register_buffer('joint_template', joint_template)

        shapedirs = self.shapedirs
        joint_shapedirs = lbs.vertices2joints(
            self.J_regressor, shapedirs.permute(2, 0, 1)).permute(1, 2, 0)
        self.register_buffer('joint_shapedirs', joint_shapedirs)

        self.shapedirs = self.shapedirs[vertex_indices]

        num_vertices = len(self.v_template)
        self.v_template = self.v_template[vertex_indices]

        selected_posedirs = self.posedirs.t().reshape(
            num_vertices, 3, -1)[vertex_indices]
        self.posedirs = selected_posedirs.reshape(
            -1, selected_posedirs.shape[-1]).t()

        self.lbs_weights = self.lbs_weights[vertex_indices]

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        pose2rot: bool = True,
        v_template: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> SMPLOutput:
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        model_vars = [betas, global_orient, body_pose, transl]
        batch_size = batch_size_from_tensor_list(model_vars)
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        targs = {'dtype': dtype, 'device': device}

        if global_orient is None:
            global_orient = identity_rot_mats(
                batch_size=batch_size, num_matrices=1, **targs)
        if body_pose is None:
            body_pose = identity_rot_mats(
                batch_size=batch_size, num_matrices=self.NUM_BODY_JOINTS,
                **targs)

        if global_orient is None:
            global_orient = torch.eye(3, **targs).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        # Concatenate all pose vectors
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3, 3),
             body_pose.reshape(batch_size, -1, 3, 3),
             ],
            dim=1)
        # shape_components = torch.cat([betas, expression], dim=-1)
        # shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)
        # shape_components = torch.cat([betas, expression], dim=-1)

        joints_shaped = self.joint_template + lbs.blend_shapes(
            betas, self.joint_shapedirs
        )
        num_joints = joints_shaped.shape[1]

        v_shaped = self.v_template + lbs.blend_shapes(
            betas, self.shapedirs)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        # rot_mats = lbs.batch_rodrigues(full_pose.view(-1, 3)).view(
        #     [batch_size, -1, 3, 3])
        rot_mats = full_pose.view(batch_size, -1, 3, 3)

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        joints, rel_transforms, abs_transforms = lbs.batch_rigid_transform(
            rot_mats, joints_shaped, self.parents,
            parallel_exec=self.parallel_exec,
            task_group_parents=self.task_group_parents,
        )

        # 5. Do skinning:
        # W is N x V x (J + 1)
        # W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        # T = torch.matmul(W, rel_transforms.view(batch_size, num_joints, 16)).view(
        #     batch_size, -1, 4, 4)
        T = torch.einsum('vj,bjmn->bvmn', [self.lbs_weights, rel_transforms])

        # homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], **targs)
        # v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        # v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        v_homo = torch.matmul(
            T, F.pad(v_posed, [0, 1], value=1).unsqueeze(dim=-1))

        vertices = v_homo[:, :, :3, 0]

        output = SMPLOutput(vertices=vertices if return_verts else None,
                            joints=joints,
                            betas=betas,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            v_shaped=v_shaped,
                            full_pose=full_pose if return_full_pose else None,
                            transl=transl,
                            )
        return output

