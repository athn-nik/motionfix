
import numpy as np
from src.info.joints import smplh_joints
left_foot_joints = []
right_foot_joints = []
jointnames = ['foot', 'small_toe', 'heel', 'big_toe', 'ankle']

def foot_detect(positions, thres):
    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, left_foot_joints, 0] - positions[:-1, left_foot_joints, 0]) ** 2
    feet_l_y = (positions[1:, left_foot_joints, 1] - positions[:-1, left_foot_joints, 1]) ** 2
    feet_l_z = (positions[1:, left_foot_joints, 2] - positions[:-1, left_foot_joints, 2]) ** 2
    #     feet_l_h = positions[:-1,fid_l,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, right_foot_joints, 0] - positions[:-1, right_foot_joints, 0]) ** 2
    feet_r_y = (positions[1:, right_foot_joints, 1] - positions[:-1, right_foot_joints, 1]) ** 2
    feet_r_z = (positions[1:, right_foot_joints, 2] - positions[:-1, right_foot_joints, 2]) ** 2
    #     feet_r_h = positions[:-1,fid_r,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r
