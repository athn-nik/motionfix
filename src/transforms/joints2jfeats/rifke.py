from typing import Optional

import torch
from einops import rearrange
from torch import Tensor
from .tools import get_forward_direction, get_floor, gaussian_filter1d  # noqa
from src.tools.geometry import matrix_of_angles, _axis_angle_rotation as axis_angle_rotation
from .base import Joints2Jfeats

import src.tools.geometry as geometry


class Rifke(Joints2Jfeats):
    def __init__(self, jointstype: str = "smplh",
                 path: Optional[str] = None,
                 normalization: bool = False,
                 forward_filter: bool = False,
                 keep_dim: bool = False,
                 **kwargs) -> None:

        if jointstype != "smplh":
            raise NotImplementedError("This jointstype is not implemented.")

        super().__init__(path=path, normalization=normalization)
        self.jointstype = jointstype
        self.forward_filter = forward_filter
        self.keep_dim = keep_dim

    def forward(self, joints: Tensor) -> Tensor:
        # Joints to rotation invariant poses (Holden et. al.)
        # Similar function than fke2rifke in Language2Pose repository
        # Adapted to pytorch
        # Put the origin center of the root joint instead of the ground projection
        poses = joints.clone()
        poses[..., 2] -= get_floor(poses, jointstype=self.jointstype)

        translation = poses[..., 0, :].clone()

        # Let the root have the Z translation --> gravity axis
        root_grav_axis = translation[..., 2]

        # Trajectory => Translation without gravity axis (Y)
        trajectory = translation[..., [0, 1]]

        # Compute the forward direction (before removing a joint)
        forward = get_forward_direction(poses, jointstype=self.jointstype)

        # Delete the root joints of the poses
        poses = poses[..., 1:, :]

        # Remove the trajectory of the poses
        poses[..., [0, 1]] -= trajectory[..., None, :]

        # Compute the trajectory
        vel_trajectory = torch.diff(trajectory, dim=-2)
        # 0 for the first one => keep the dimentionality
        vel_trajectory = torch.cat((0 * vel_trajectory[..., [0], :], vel_trajectory), dim=-2)

        if self.forward_filter:
            # Smoothing to remove high frequencies
            forward = gaussian_filter1d(forward, 2)
            # normalize again to get real directions
            forward = torch.nn.functional.normalize(forward, dim=-1)

        # changed this also for New pytorch
        angles = torch.atan2(*(forward.transpose(0, -1))).transpose(0, -1)


        # replace the diff
        # vel_angles = torch.diff(angles, dim=-1)
        # because it can be wrong
        mat_angles = axis_angle_rotation("Z", angles)
        diff_mat_rotZ = mat_angles[..., 1:, :, :] @ mat_angles.transpose(-1, -2)[..., :-1, :, :]
        vel_angles = geometry.matrix_to_axis_angle(diff_mat_rotZ)[..., 2]

        # 0 for the first one => keep the dimentionality
        vel_angles = torch.cat((0 * vel_angles[..., [0]], vel_angles), dim=-1)

        # Construct the inverse rotation matrix
        sin, cos = forward[..., 0], forward[..., 1]
        rotations_inv = matrix_of_angles(cos, sin, inv=True)
        # Rotate the poses
        poses_local = torch.einsum("...lj,...jk->...lk", poses[..., [0, 1]], rotations_inv)
        poses_local = torch.stack((poses_local[..., 0], poses_local[..., 1], poses[..., 2]), axis=-1)

        if not self.keep_dim:
            # stack the xyz joints into feature vectors
            poses_features = rearrange(poses_local, "... joints xyz -> ... (joints xyz)")

        # Rotate the vel_trajectory
        vel_trajectory_local = torch.einsum("...j,...jk->...k", vel_trajectory, rotations_inv)

        number_dim = len(root_grav_axis.shape)  # should be 1 if single or 2 if batched
        tiling_tuple = tuple([1 for _ in range(number_dim)] + [3])
        root_grav_3 = torch.tile(root_grav_axis[..., None], tiling_tuple)

        vels_3 = torch.cat((vel_angles[..., None], vel_trajectory_local), -1)

        # root_y_3[..., None, :]  => torch.Size([.., 67, 1, 3])
        # poses_local     =>         torch.Size([.., 67, 20, 3])
        # vels_3[..., None, :]    => torch.Size([.., 67, 1, 3])

        if self.keep_dim:
            # Stack things together
            features = torch.cat((root_grav_3[..., None, :],
                                  poses_local,
                                  vels_3[..., None, :]), -2)
        else:
            # Stack things together
            features = torch.cat((root_grav_axis[..., None],
                                  poses_features,
                                  vel_angles[..., None],
                                  vel_trajectory_local), -1)

        # Normalize if needed
        features = self.normalize(features)
        return features

    def inverse(self, features: Tensor) -> Tensor:
        features = self.unnormalize(features)
        root_grav_axis, poses_features, vel_angles, vel_trajectory_local = self.extract(features)

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the poses
        poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 1]], rotations)
        poses = torch.stack((poses[..., 0], poses[..., 1], poses_local[..., 2]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)

        # put back the root joint y
        poses[..., 0, 2] = root_grav_axis

        # Add the trajectory globally
        poses[..., [0, 1]] += trajectory[..., None, :]
        return poses

    def extract(self, features: Tensor) -> tuple[Tensor]:
        root_grav_axis = features[..., 0]
        poses_features = features[..., 1:-3]
        vel_angles = features[..., -3]
        vel_trajectory_local = features[..., -2:]

        return root_grav_axis, poses_features, vel_angles, vel_trajectory_local
