from pytorch3d.transforms import (
    axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix,
    matrix_to_axis_angle, euler_angles_to_matrix, matrix_to_euler_angles
)
from einops import rearrange
import numpy as np
import torch
from torch import Tensor
from roma import rotmat_to_rotvec, rotvec_to_rotmat
from torch.nn.functional import pad

def transform_body_pose(pose, formats):
    """
    various angle transformations, transforms input to torch.Tensor
    input:
        - pose: pose tensor
        - formats: string denoting the input-output angle format
    """
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    if formats == "6d->aa":
        j = pose.shape[-1] / 6
        pose = rearrange(pose, '... (j d) -> ... j d', d=6)
        pose = pose.squeeze(-2)  # in case of only one angle
        pose = rotation_6d_to_matrix(pose)
        pose = matrix_to_axis_angle(pose)
        if j > 1:
            pose = rearrange(pose, '... j d -> ... (j d)')
    elif formats == "aa->6d":
        j = pose.shape[-1] / 3
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = pose.squeeze(-2)  # in case of only one angle
        # axis-angle to rotation matrix & drop last row
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
        if j > 1:
            pose = rearrange(pose, '... j d -> ... (j d)')
    elif formats == "aa->rot":
        j = pose.shape[-1] / 3
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = pose.squeeze(-2)  # in case of only one angle
        # axis-angle to rotation matrix & drop last row
        pose = torch.clamp(axis_angle_to_matrix(pose), min=-1.0, max=1.0)
    elif formats == "6d->rot":
        j = pose.shape[-1] / 6
        pose = rearrange(pose, '... (j d) -> ... j d', d=6)
        pose = pose.squeeze(-2)  # in case of only one angle
        pose = torch.clamp(rotation_6d_to_matrix(pose), min=-1.0, max=1.0)
    elif formats == "rot->aa":
        # pose = rearrange(pose, '... (j d1 d2) -> ... j d1 d2', d1=3, d2=3)
        pose = matrix_to_axis_angle(pose)
    elif formats == "rot->6d":
        # pose = rearrange(pose, '... (j d1 d2) -> ... j d1 d2', d1=3, d2=3)
        pose = matrix_to_rotation_6d(pose)
    else:
        raise ValueError(f"specified conversion format is invalid: {formats}")
    return pose

def apply_rot_delta(rots, deltas, in_format="6d", out_format="6d"):
    """
    rots needs to have same dimentionality as delta
    """
    assert rots.shape == deltas.shape
    if in_format == "aa":
        j = rots.shape[-1] / 3
    elif in_format == "6d":
        j = rots.shape[-1] / 6
    else:
        raise ValueError(f"specified conversion format is unsupported: {in_format}")
    rots = transform_body_pose(rots, f"{in_format}->rot")
    deltas = transform_body_pose(deltas, f"{in_format}->rot")
    new_rots = torch.einsum("...ij,...jk->...ik", rots, deltas)  # Ri+1=Ri@delta
    new_rots = transform_body_pose(new_rots, f"rot->{out_format}")
    if j > 1:
        new_rots = rearrange(new_rots, '... j d -> ... (j d)')
    return new_rots

def rot_diff(rots1, rots2=None, in_format="6d", out_format="6d"):
    """
    dim 0 is considered to be the time dimention, this is where the shift will happen
    """
    if in_format == "aa":
        j = rots1.shape[-1] / 3
    elif in_format == "6d":
        j = rots1.shape[-1] / 6
    else:
        raise ValueError(f"specified conversion format is unsupported: {in_format}")
    rots1 = transform_body_pose(rots1, f"{in_format}->rot")
    if rots2 is not None:
        rots2 = transform_body_pose(rots2, f"{in_format}->rot")
    else:
        rots2 = rots1
        rots1 =  rots1.roll(1, 0)
        
    rots_diff = torch.einsum("...ij,...ik->...jk", rots1, rots2)  # Ri.T@R_i+1
    if rots2 is None:
        rots_diff[0, ..., :, :] = torch.eye(3, device=rots1.device)
    rots_diff = transform_body_pose(rots_diff, f"rot->{out_format}")
    if j > 1:
        rots_diff = rearrange(rots_diff, '... j d -> ... (j d)')
    return rots_diff

def change_for(p, R, T=0, forward=True):
    """
    Change frame of reference for vector p
    p: vector in original coordinate frame
    R: rotation matrix of new coordinate frame ([x, y, z] format)
    T: translation of new coordinate frame
    """
    if forward:  # R.T @ (p_global - pelvis_translation)
        return torch.einsum('...di,...d->...i', R, p - T)
    else:  # R @ (p_global - pelvis_translation)
        return torch.einsum('...di,...i->...d', R, p) + T

def get_z_rot(rot_, in_format="6d"):
    rot = rot_.clone().detach()
    rot = transform_body_pose(rot, f"{in_format}->rot")
    euler_z = matrix_to_euler_angles(rot, "ZYX")
    euler_z[..., 1:] = 0.0
    z_rot = torch.clamp(
        euler_angles_to_matrix(euler_z, "ZYX"),
        min=-1.0, max=1.0)  # add zero XY euler angles
    return z_rot
    

def remove_z_rot(pose, in_format="6d", out_format="6d"):
    """
    zero-out the global orientation around Z axis
    """
    assert out_format == "6d"
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    # transform to matrix
    pose = transform_body_pose(pose, f"{in_format}->rot")
    pose = matrix_to_euler_angles(pose, "ZYX")
    pose[..., 0] = 0
    pose = matrix_to_rotation_6d(torch.clamp(
        euler_angles_to_matrix(pose, "ZYX"),
        min=-1.0, max=1.0))
    return pose

def local_to_global_orient(body_orient: Tensor, poses: Tensor, parents: list,
                           input_format='aa', output_format='aa'):
    """
    Modified from aitviewer
    Convert relative joint angles to global by unrolling the kinematic chain.
    This function is fully differentiable ;)
    :param poses: A tensor of shape (N, N_JOINTS*d) defining the relative poses in angle-axis format.
    :param parents: A list of parents for each joint j, i.e. parent[j] is the parent of joint j.
    :param output_format: 'aa' for axis-angle or 'rotmat' for rotation matrices.
    :param input_format: 'aa' or 'rotmat' ...
    :return: The global joint angles as a tensor of shape (N, N_JOINTS*DOF).
    """
    assert output_format in ['aa', 'rotmat']
    assert input_format in ['aa', 'rotmat']
    dof = 3 if input_format == 'aa' else 9
    n_joints = poses.shape[-1] // dof + 1
    if input_format == 'aa':
        body_orient = rotvec_to_rotmat(body_orient)
        local_oris = rotvec_to_rotmat(rearrange(poses, '... (j d) -> ... j d', d=3))
        local_oris = torch.cat((body_orient[..., None, :, :], local_oris), dim=-3)
    else:
        # this part has not been tested
        local_oris = torch.cat((body_orient[..., None, :, :], local_oris), dim=-3)
    global_oris_ = []

    # Apply the chain rule starting from the pelvis
    for j in range(n_joints):
        if parents[j] < 0:
            # root
            global_oris_.append(local_oris[..., j, :, :])
        else:
            parent_rot = global_oris_[parents[j]]
            local_rot = local_oris[..., j, :, :]
            global_oris_.append(torch.einsum('...ij,...jk->...ik', parent_rot, local_rot))
            # global_oris[..., j, :, :] = torch.bmm(parent_rot, local_rot)
    global_oris = torch.stack(global_oris_, dim=1)
    # global_oris: ... x J x 3 x 3
    # account for the body's root orientation
    # global_oris = torch.einsum('...ij,...jk->...ik', body_orient[..., None, :, :], global_oris)

    if output_format == 'aa':
        return rotmat_to_rotvec(global_oris)
        # res = global_oris.reshape((-1, n_joints * 3))
    else:
        return global_oris
    # return res