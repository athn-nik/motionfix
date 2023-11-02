from pathlib import Path
import torch
name_mapping = {'MPI_mosh':'MoSh',
                'ACCAD':'ACCAD',
                'WEIZMANN': 'WEIZMANN',
                'CNRS': 'CNRS',
                'DFaust': 'DFaust',
                'TCDHands': 'TCDHands',
                'SOMA': 'SOMA',
                'CMU': 'CMU',
                'SFU': 'SFU',
                'TotalCapture': 'TotalCapture',
                'HDM05': 'HDM05',
                'Transitions': 'Transitions',
                'MoSh': 'MoSh',
                'KIT': 'KIT',
                'DanceDB': 'DanceDB',
                'Transitions_mocap': 'Transitions',
                'PosePrior': 'PosePrior',
                'MPI_Limits': 'PosePrior',
                'BMLhandball': 'BMLhandball',
                'SSM': 'SSM',
                'TCD_handMocap': 'TCDHands',
                'BMLrub': 'BMLrub',
                'BioMotionLab_NTroje': 'BMLrub',
                'SSM_synced': 'SSM',
                'Eyes_Japan_Dataset': 'Eyes_Japan_Dataset',
                'DFaust_67': 'DFaust',
                'EKUT': 'EKUT',
                'MPI_HDM05': 'HDM05',
                'GRAB': 'GRAB',
                'HumanEva': 'HumanEva',
                'HUMAN4D': 'HUMAN4D',
                'BMLmovi': 'BMLmovi'
                }


def path_normalizer(paths_list_or_str):
    if isinstance(paths_list_or_str, str):
        paths_list = [paths_list_or_str]
    else:
        paths_list = list(paths_list_or_str)
    # works only for dir1/dir2/fname.npz to normalize dir1
    plist = ['/'.join(p.split('/')[-3:]) for p in paths_list]
    norm_path = ['/'.join([name_mapping[p.split('/')[0]], p.split('/')[1], p.split('/')[2]]) for p in plist if p.split('/')[0] in name_mapping.keys()]
    if isinstance(paths_list_or_str, str):
        return norm_path[0]
    else:
        return norm_path

def fname_normalizer(fname):
    dataset_name, subject, sequence_name = fname.split('/')
    sequence_name = sequence_name.replace('_poses.npz', '')
    sequence_name = sequence_name.replace('_poses', '')
    sequence_name = sequence_name.replace('poses', '')
    sequence_name = sequence_name.replace('_stageii.npz', '')
    sequence_name = sequence_name.replace('_stageii', '')
    sequence_name = sequence_name.rstrip()
    getVals = list([val for val in sequence_name
            if val.isalpha() or val.isnumeric()])
    sequence_name = ''.join(getVals)

    return '/'.join([dataset_name, subject, sequence_name])

def flip_motion(pose, trans):
    #  expects T, Jx3
    # Permutation of SMPL pose parameters when flipping the shape
    SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10,
                             12, 14, 13, 15, 17, 16, 19,
                              18, 21, 20] #, 23, 22]
    SMPL_POSE_FLIP_PERM = []
    for i in SMPL_JOINTS_FLIP_PERM:
        SMPL_POSE_FLIP_PERM.append(3*i)
        SMPL_POSE_FLIP_PERM.append(3*i+1)
        SMPL_POSE_FLIP_PERM.append(3*i+2)

    flipped_parts = SMPL_POSE_FLIP_PERM
    pose = pose[:, flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[:, 1::3] = -pose[:, 1::3]
    pose[:, 2::3] = -pose[:, 2::3]
    x, y, z = torch.unbind(trans, dim=-1)
    mirrored_trans = torch.stack((-x, y, z), axis=-1) 

    return pose, mirrored_trans