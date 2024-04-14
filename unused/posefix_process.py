##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import json
import roma
import torch
import numpy as np

# this code comes from:
# https://github.com/naver/posescript/blob/main/src/text2pose/utils.py
# https://github.com/naver/posescript/blob/main/src/text2pose/data.py
# https://github.com/naver/posescript/blob/main/src/text2pose/config.py


## INPUT
################################################################################

POSEFIX_LOCATION = 'data/posefix_release' # TODO
# download data from: https://download.europe.naverlabs.com/ComputerVision/PoseFix/posefix_dataset_release.zip
split = "train" # TODO

# choose what texts you want # TODO
caption_files = [
     f"{POSEFIX_LOCATION}/posefix_human_6157.json", 
    #  f"{POSEFIX_LOCATION}/posefix_paraphrases_4284.json",
    #  f"{POSEFIX_LOCATION}/posefix_auto_135305.json"
     ]

AMASS_FILE_LOCATION = f"/ps/project/motion_text_synthesis/amass/smplhg" # TODO

## SETUP
################################################################################

supported_datasets = {"AMASS":AMASS_FILE_LOCATION}
file_pose_id_2_dataset_sequence_and_frame_index = f"{POSEFIX_LOCATION}/ids_2_dataset_sequence_and_frame_index_100k.json" 
file_pair_id_2_pose_ids = f"{POSEFIX_LOCATION}/pair_id_2_pose_ids.json"
file_posefix_split = f"{POSEFIX_LOCATION}/%s_%s_sequence_pair_ids.json" # %s %s --> (train|val|test), (in|out)


## FUNCTIONS & BUILD TRIPLET DATA
################################################################################

def read_json(absolute_filepath):
	with open(absolute_filepath, 'r') as f:
		data = json.load(f)
	return data


def load_posefix(caption_files, split, dataID_2_pose_info):
    
    # initialize split ids (pair ids) ; split dependent
    dataIDs = read_json(file_posefix_split % (split, 'in'))
    dataIDs += read_json(file_posefix_split % (split, 'out'))
    
    # load pose pairs (pairs of pose ids)
    pose_pairs = read_json(file_pair_id_2_pose_ids)

    # initialize triplet data
    triplets = {data_id: {"pose_A": pose_pairs[data_id][0],
                          "pose_B": pose_pairs[data_id][1],
                          "modifier": []}
                    for data_id in dataIDs}
    for t,v in triplets.items(): triplets[t]["in-sequence"] = dataID_2_pose_info[str(v["pose_A"])][1] == dataID_2_pose_info[str(v["pose_B"])][1]

    # load available modifiers
    for triplet_file in caption_files:
        annotations = read_json(triplet_file)
        for data_id_str, c in annotations.items():
            try:
                triplets[int(data_id_str)]["modifier"] += c if type(c) is list else [c]
            except KeyError:
                # this annotation is not part of the split
                pass

    # clean dataIDs and triplets: remove access to unavailable
    # data (when no annotation was performed); necessary step if a smaller
    # data set was loaded
    dataIDs = [data_id for data_id in dataIDs if triplets[data_id]["modifier"]]
    triplets = {data_id:triplets[data_id] for data_id in dataIDs}
    
    duplicates = 0
    for t in triplets.values():
        duplicates += 1 if len(t["modifier"]) > 1 else 0
    print(f"[PoseFix] Loaded {len(dataIDs)} pairs in {split} split (found {duplicates} with more than 1 annotation).")
    
    return dataIDs, triplets


dataID_2_pose_info = read_json(file_pose_id_2_dataset_sequence_and_frame_index)
dataIDs, triplets = load_posefix(caption_files, split, dataID_2_pose_info)


def rotvec_to_eulerangles(x):
	x_rotmat = roma.rotvec_to_rotmat(x)
	thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
	thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
	thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
	return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
	N = thetax.numel()
	# rotx
	rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotx[:,1,1] = torch.cos(thetax)
	rotx[:,2,2] = torch.cos(thetax)
	rotx[:,1,2] = -torch.sin(thetax)
	rotx[:,2,1] = torch.sin(thetax)
	roty[:,0,0] = torch.cos(thetay)
	roty[:,2,2] = torch.cos(thetay)
	roty[:,0,2] = torch.sin(thetay)
	roty[:,2,0] = -torch.sin(thetay)
	rotz[:,0,0] = torch.cos(thetaz)
	rotz[:,1,1] = torch.cos(thetaz)
	rotz[:,0,1] = -torch.sin(thetaz)
	rotz[:,1,0] = torch.sin(thetaz)
	rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
	return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
	rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
	return roma.rotmat_to_rotvec(rotmat)

def get_pose_data_from_file(pose_info, applied_rotation=None, output_rotation=False):
	"""
	Load pose data and normalize the orientation.

	Args:
		pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]
		applied_rotation: rotation to be applied to the pose data. If None, the
			normalization rotation is applied.
		output_rotation: whether to output the rotation performed for
			normalization, in addition of the normalized pose data.

	Returns:
		pose data, torch.tensor of size (1, n_joints*3), all joints considered.
		(optional) R, torch.tensor representing the rotation of normalization
	"""

	# load pose data
	assert pose_info[0] in supported_datasets, f"Expected data from on of the following datasets: {','.join(supported_datasets)} (provided dataset: {pose_info[0]})."
	
	if pose_info[0] == "AMASS":
		dp = np.load(os.path.join(supported_datasets[pose_info[0]], pose_info[1]))
		pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
		pose = torch.as_tensor(pose).to(dtype=torch.float32)

	# normalize the global orient
	initial_rotation = pose[:1,:].clone()
	if applied_rotation is None:
		thetax, thetay, thetaz = rotvec_to_eulerangles( initial_rotation )
		zeros = torch.zeros_like(thetaz)
		pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)
	else:
		pose[0:1,:] = roma.rotvec_composition((applied_rotation, initial_rotation))
	if output_rotation:
		# a = A.u, after normalization, becomes a' = A'.u
		# we look for the normalization rotation R such that: a' = R.a
		# since a = A.u ==> u = A^-1.a
		# a' = A'.u = A'.A^-1.a ==> R = A'.A^-1
		R = roma.rotvec_composition((pose[0:1,:], roma.rotvec_inverse(initial_rotation)))
		return pose.reshape(1, -1), R
	
	return pose.reshape(1, -1)


def get_pose(index, pose_type, applied_rotation=None, output_rotation=False):
    # get pose id
    pose_id = triplets[dataIDs[index]][pose_type]
    # load pose data
    pose_info = dataID_2_pose_info[str(pose_id)]
    ret = get_pose_data_from_file(pose_info,
                                        applied_rotation=applied_rotation,
                                        output_rotation=output_rotation)
    # reshape pose to (njoints, 3)
    if output_rotation:
        return ret[0].reshape(-1, 3), int(pose_id), ret[1] # rotation
    else:
        return ret.reshape(-1, 3), int(pose_id)
        

def get_poses_AB(index):
    if triplets[dataIDs[index]]["in-sequence"]:
        pA, pidA, rA = get_pose(index, pose_type="pose_A", output_rotation=True)
        pB, pidB = get_pose(index, pose_type="pose_B", applied_rotation=rA)
    else:
        pA, pidA = get_pose(index, pose_type="pose_A")
        pB, pidB = get_pose(index, pose_type="pose_B")
    return pA, pidA, pB, pidB


for index in range(len(dataIDs)):
    pA, _, pB, _ = get_poses_AB(index)
    triplets[dataIDs[index]].update({
         "pose_A_data": pA,
         "pose_B_data": pB,
    })


# import ipdb; ipdb.set_trace()
import joblib
joblib.dump(triplets, 'data/posefix_release/posefix_d.pth.tar')
# TODO: save `triplets` in your favorite format