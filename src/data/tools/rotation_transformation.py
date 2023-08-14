import torch
import src.tools.geometry as geometry
from einops import rearrange


def rotate_trajectory(traj, rotZ, inverse=False):
    if inverse:
        # transpose
        rotZ = rearrange(rotZ, "... i j -> ... j i")

    vel = torch.diff(traj, dim=-2)
    # 0 for the first one => keep the dimentionality
    vel = torch.cat((0 * vel[..., [0], :], vel), dim=-2)
    vel_local = torch.einsum("...kj,...k->...j", rotZ[..., :2, :2], vel[..., :2])
    # Integrate the trajectory
    traj_local = torch.cumsum(vel_local, dim=-2)
    # First frame should be the same as before
    traj_local = traj_local - traj_local[..., [0], :] + traj[..., [0], :]
    return traj_local


def rotate_trans(trans, rotZ, inverse=False):
    traj = trans[..., :2]
    transZ = trans[..., 2]
    traj_local = rotate_trajectory(traj, rotZ, inverse=inverse)
    trans_local = torch.cat((traj_local, transZ[..., None]), axis=-1)
    return trans_local


def rotate_joints2D(joints, rotZ, inverse=False):
    if inverse:
        # transpose
        rotZ = rearrange(rotZ, "... i j -> ... j i")

    assert joints.shape[-1] == 2
    vel = torch.diff(joints, dim=-2)
    # 0 for the first one => keep the dimentionality
    vel = torch.cat((0 * vel[..., [0], :], vel), dim=-2)
    vel_local = torch.einsum("...kj,...lk->...lj", rotZ[..., :2, :2], vel[..., :2])
    # Integrate the trajectory
    joints_local = torch.cumsum(vel_local, dim=-2)
    # First frame should be the same as before
    joints_local = joints_local - joints_local[..., [0], :] + joints[..., [0], :]
    return joints_local


def rotate_joints(joints, rotZ, inverse=False):
    joints2D = joints[..., :2]
    jointsZ = joints[..., 2]
    joints2D_local = rotate_joints2D(joints2D, rotZ, inverse=inverse)
    joints_local = torch.cat((joints2D_local, jointsZ[..., None]), axis=-1)
    return joints_local


def canonicalize_rotations(global_orient, trans, angle=0.0):
    global_euler = geometry.matrix_to_euler_angles(global_orient, "ZYX")
    anglesZ, anglesY, anglesX = torch.unbind(global_euler, -1)

    rotZ = geometry._axis_angle_rotation("Z", anglesZ)

    # remove the current rotation
    # make it local
    local_trans = rotate_trans(trans, rotZ)

    # For information:
    # rotate_joints(joints, rotZ) == joints_local

    diff_mat_rotZ = rotZ[..., 1:, :, :] @ rotZ.transpose(-1, -2)[..., :-1, :, :]

    vel_anglesZ = geometry.matrix_to_axis_angle(diff_mat_rotZ)[..., 2]
    # padding "same"
    vel_anglesZ = torch.cat((vel_anglesZ[..., [0]], vel_anglesZ), dim=-1)

    # Compute new rotation:
    # canonicalized
    anglesZ = torch.cumsum(vel_anglesZ, -1)
    anglesZ += angle
    rotZ = geometry._axis_angle_rotation("Z", anglesZ)

    new_trans = rotate_trans(local_trans, rotZ, inverse=True)

    new_global_euler = torch.stack((anglesZ, anglesY, anglesX), -1)
    new_global_orient = geometry.euler_angles_to_matrix(new_global_euler, "ZYX")

    return new_global_orient, new_trans