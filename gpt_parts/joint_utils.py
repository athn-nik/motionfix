import string

mmm_joints = [
    "root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
    "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"
]

number_of_joints = {
    "smplh": 22,
}

smplh_joints = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_index1", "left_index2", "left_index3", "left_middle1",
    "left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
    "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
    "left_thumb2", "left_thumb3", "right_index1", "right_index2",
    "right_index3", "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2",
    "right_thumb3", "nose", "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle",
    "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle",
    "right_ring", "right_pinky"
]

smpl_bps = {
    'global': ['pelvis'],
    'torso': ['spine1', 'spine2', 'spine3', 'neck', 'head'],
    'left arm': ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist'],
    'right arm':
    ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist'],
    'left leg': ['left_hip', 'left_knee', 'left_ankle', 'left_foot'],
    'right leg': ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
}

bp2ids = {
    bp_name: [smplh_joints.index(j) for j in jts_names]
    for bp_name, jts_names in smpl_bps.items()
    }

mmm2smplh_correspondence = {
    "root": "pelvis",
    "BP": "spine1",
    "BT": "spine3",
    "BLN": "neck",
    "BUN": "head",
    "LS": "left_shoulder",
    "LE": "left_elbow",
    "LW": "left_wrist",
    "RS": "right_shoulder",
    "RE": "right_elbow",
    "RW": "right_wrist",
    "LH": "left_hip",
    "LK": "left_knee",
    "LA": "left_ankle",
    "LMrot": "left_heel",
    "LF": "left_foot",
    "RH": "right_hip",
    "RK": "right_knee",
    "RA": "right_ankle",
    "RMrot": "right_heel",
    "RF": "right_foot"
}
smplh2mmm_correspondence = {
    val: key
    for key, val in mmm2smplh_correspondence.items()
}

smplh2mmm_indexes = [
    smplh_joints.index(mmm2smplh_correspondence[x]) for x in mmm_joints
]

smpl2gpt = {
    'global': [
        'spine', 'butt', 'buttocks', 'buttock', 'crotch', 'pelvis', 'groin',
        'bottom', 'waist',
    ],
    'torso': [
        'spine', 'body', 'head', 'neck', 'torso', 'trunk', 'jaw', 'nose',
        'breast', 'chest', 'belly', 'mouth', 'throat',
        'chin', 'chest', 'back', 'face',
        'jaws', 'side', 'teeth'
    ],
    'left arm': [
        'arms', 'hands', 'shoulders', 'elbows', 'arm', 'wrists', 'bicep',
        'palm',   'wrist', 'shoulder', 'hand', 'arm', 'elbow',
        'tricep',  'biceps', 'thumb', 'fists', 'finger', 'fingers',
        'deltoid', 'trapezius', 
    ],
    'right arm': [
        'arms', 'hands', 'shoulders', 'elbows', 'arm', 'wrists', 'bicep',
        'palm',  'wrist', 'shoulder', 'hand', 'arm', 'elbow', 'tricep',  'biceps',
        'thumb', 'fists', 'finger', 'fingers', 'deltoid', 
    ],
    'left leg': [
        'legs', 'feet', 'hips', 'knee', 'ankle', 'leg', 'hip', 'calf',
        'thigh', 'thighs', 'foot', 'knees', 'ankles', 'heel', 
         'toe', 'toes', 'calves'
    ],
    'right leg': [
        'legs', 'feet', 'hips', 'knee', 'ankle', 'leg', 'hip', 'calf',
        'thigh', 'thighs', 'foot', 'knees', 'ankles', 'heel',
         'toe', 'toes',  'calves'
    ]
}

body_parts = ['left arm', 'right arm', 'left leg', 'global orientation',
'right leg', 'torso', 'left hand', 'right hand', 'left ankle', 'right ankle', 'left foot',
'right foot', 'head', 'neck', 'right shoulder', 'left shoulder', 'pelvis', 'spine']

body_parts_coarse = ['arm', 'arms', 'leg', 'legs' 'right arm', 'left leg', 'global orientation',
'right leg', 'torso', 'left hand', 'right hand', 'left ankle', 'right ankle', 'left foot',
'right foot', 'head', 'neck', 'right shoulder', 'left shoulder', 'pelvis', 'spine']

def get_bps_from_gpt(gpt_ans):
    cur_lbl = gpt_ans
    cur_lbl = cur_lbl.translate(str.maketrans('', '', string.punctuation)).lower().replace('\n', ' ')
    cur_lbl = cur_lbl.split(' ')
    bps = [bp for bp, wds_bp in smpl2gpt.items() if set(cur_lbl) & set(wds_bp)]
    precise_bp = bps
    if [phr for phr in gpt_side['right'] if phr in cur_lbl]:
        precise_bp = [x for x in bps if not x.startswith('left')]
    elif [phr for phr in gpt_side['left'] if phr in cur_lbl]:
        precise_bp = [x for x in bps if not x.startswith('right')]
    bp_list = [0] * len(smpl_bps_ids)
    for bp_str in precise_bp:
        bp_list[smpl_bps_ids[bp_str]] += 1
    return bp_list

def get_bp_from_gpt_list(text, gpt_labels_full_sent, return_original=False):
    if text in ['animal behavior series',]:
        if return_original:
            return [1] * len(smpl_bps_ids), gpt_labels_full_sent[text][2]

        return [1] * len(smpl_bps_ids)
    else:
        original_cur_lbl = gpt_labels_full_sent[text][2]
        cur_lbl = original_cur_lbl.translate(str.maketrans('', '',
                                                        string.punctuation))
        cur_lbl = cur_lbl.lower().replace('answer', '')
        precise_bp = cur_lbl.strip().split('\n')

        bp_list = [0] * len(smpl_bps_ids)
        
        for bp_str in precise_bp:
            if bp_str == 'buttocks' or bp_str=='waist':
                bp_final = 'global'
            elif bp_str == 'neck':
                bp_final = 'torso'
            else:
                bp_final = str(bp_str)
            try:
                bp_list[smpl_bps_ids[bp_final]] += 1
            except:
                import ipdb; ipdb.set_trace()
        if bp_list == [0, 0 ,0, 0, 0, 0]:
            import ipdb; ipdb.set_trace()
        if return_original:
            return bp_list, original_cur_lbl
        return bp_list


def get_gpt(a_t, gpt_act2bp):
    # a = read_json('deps/gpt/gpt3-labels.json')
    cur_lbl = gpt_act2bp[a_t]['GPT-response']
    cur_lbl = cur_lbl.translate(str.maketrans('', '', string.punctuation))
    cur_lbl = cur_lbl.lower().split(' ')
    bps = [bp for bp, wds_bp in smpl2gpt.items() if set(cur_lbl) & set(wds_bp)]
    precise_bp = bps
    if [phr for phr in gpt_side['right'] if phr in cur_lbl]:
        precise_bp = [x for x in bps if not x.startswith('left')]
    elif [phr for phr in gpt_side['left'] if phr in cur_lbl]:
        precise_bp = [x for x in bps if not x.startswith('right')]
    bp_list = [0] * len(smpl_bps_ids)
    for bp_str in precise_bp:
        bp_list[smpl_bps_ids[bp_str]] += 1
    return bp_list


def get_compat_matrix(gpt_act2bp):
    import numpy as np
    from sinc.tools.frank import text_list_to_bp
    all_actions = list(gpt_act2bp.keys())
    act_combs = {}
    actions_bps = np.array([text_list_to_bp(x, gpt_act2bp) for x in all_actions]) # get_gpt before
    for act, bp_label in zip(all_actions, actions_bps):
        cur = np.tile(bp_label, (len(all_actions), 1))
        compat_act = cur * actions_bps
        idxs = np.where(np.all(compat_act == 0, axis=1))[0]
        act_combs[act] = [all_actions[x] for x in idxs]
    return act_combs


smpl_bps_ids = {
    'global': 0,
    'torso': 1,
    'left arm': 2,
    'right arm': 3,
    'left leg': 4,
    'right leg': 5
}

smpl_bps_ids_list = ['global', 'torso', 'left arm',
                     'right arm', 'left leg', 'right leg']

gpt_side = {
    'right': ['on the right', 'on the right side', 'right'],
    'left': ['on the left', 'on the left side', 'left']
}


def smplh2bps(motion_feats, inverse=False):
    import torch
    from einops import rearrange
    bp2ids = {
        bp_name: [smplh_joints.index(j) for j in jts_names]
        for bp_name, jts_names in smpl_bps.items()
    }

    if inverse:
        # only smpl for now
        # for NOW THE INVERSE IS NOT EXACT BECAUSE IT FLATTENS

        bs, seqlen = motion_feats[0].shape[:2]
        device = motion_feats[0].device
        # version of list of padded stuff
        global_orient = motion_feats[0][..., 3:][..., None, :]
        grav_axis_trans = motion_feats[0][..., 0]
        trajectory_vel_xy = motion_feats[0][..., 1:3]
        pseudo_global_joint = torch.cat(
            (grav_axis_trans[..., None], trajectory_vel_xy,
             torch.zeros((bs, seqlen, 3), device=device)),
            dim=-1)[..., None, :]

        motion_feats[1] = rearrange(
            motion_feats[1],
            "bs time (bpjts reprdim) -> bs time bpjts reprdim",
            reprdim=6)
        motion_feats[2] = rearrange(
            motion_feats[2],
            "bs time (bpjts reprdim) -> bs time bpjts reprdim",
            reprdim=6)
        motion_feats[3] = rearrange(
            motion_feats[3],
            "bs time (bpjts reprdim) -> bs time bpjts reprdim",
            reprdim=6)
        motion_feats[4] = rearrange(
            motion_feats[4],
            "bs time (bpjts reprdim) -> bs time bpjts reprdim",
            reprdim=6)
        motion_feats[5] = rearrange(
            motion_feats[5],
            "bs time (bpjts reprdim) -> bs time bpjts reprdim",
            reprdim=6)
        # with padding version
        res_keep_dim = torch.cat([pseudo_global_joint] + [global_orient] +
                                 motion_feats[1:],
                                 dim=-2)

        rearranged_feats = torch.zeros(bs, seqlen, 22, 6, device=device)
        rearranged_feats[:, :, bp2ids['torso']] = motion_feats[1]
        rearranged_feats[:, :, bp2ids['left arm']] = motion_feats[2]
        rearranged_feats[:, :, bp2ids['right arm']] = motion_feats[3]
        rearranged_feats[:, :, bp2ids['left leg']] = motion_feats[4]
        rearranged_feats[:, :, bp2ids['right leg']] = motion_feats[5]
        # global orient
        rearranged_feats[:, :, 0] = motion_feats[0][..., 3:]
        res = torch.cat(
            (
                motion_feats[0][..., 0][..., None],  #
                motion_feats[0][..., 1:3],
                rearranged_feats.flatten(start_dim=-2)),
            dim=-1)
        return res, res_keep_dim
    else:
        motion_rotations = motion_feats[:, :, 1:]

        xy_vel_z_trans = motion_feats[..., 0, :3]
        global_orient = motion_rotations[:, :, bp2ids['global']]
        # squeeze to flatten!! affects batch-size if naive

        orient_traj = torch.cat((xy_vel_z_trans, global_orient.squeeze(2)),
                                dim=-1)

        la = motion_rotations[:, :, bp2ids['left arm']].flatten(start_dim=-2)
        ra = motion_rotations[:, :, bp2ids['right arm']].flatten(start_dim=-2)
        ll = motion_rotations[:, :, bp2ids['left leg']].flatten(start_dim=-2)
        rl = motion_rotations[:, :, bp2ids['right leg']].flatten(start_dim=-2)
        tor = motion_rotations[:, :, bp2ids['torso']].flatten(start_dim=-2)
        res = [orient_traj, tor, la, ra, ll, rl]
        return res


mmm_kinematic_tree = [[0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10],
                      [0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20]]

smplh_kinematic_tree = [[0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20],
                        [9, 14, 17, 19, 21], [0, 1, 4, 7, 10],
                        [0, 2, 5, 8, 11]]


mmm_joints = [
    "root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
    "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"
]

smplnh_joints = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
]

smplnh2smplh_correspondence = {key: key for key in smplnh_joints}
smplh2smplnh_correspondence = {
    val: key
    for key, val in smplnh2smplh_correspondence.items()
}

smplh2smplnh_indexes = [
    smplh_joints.index(smplnh2smplh_correspondence[x]) for x in smplnh_joints
]

smplh_to_mmm_scaling_factor = 480 / 0.75
mmm_to_smplh_scaling_factor = 0.75 / 480

mmm_joints_info = {
    "root":
    mmm_joints.index("root"),
    "feet": [
        mmm_joints.index("LMrot"),
        mmm_joints.index("RMrot"),
        mmm_joints.index("LF"),
        mmm_joints.index("RF")
    ],
    "shoulders": [mmm_joints.index("LS"),
                  mmm_joints.index("RS")],
    "hips": [mmm_joints.index("LH"),
             mmm_joints.index("RH")]
}

smplnh_joints_info = {
    "root":
    smplnh_joints.index("pelvis"),
    "feet": [
        smplnh_joints.index("left_ankle"),
        smplnh_joints.index("right_ankle"),
        smplnh_joints.index("left_foot"),
        smplnh_joints.index("right_foot")
    ],
    "shoulders": [
        smplnh_joints.index("left_shoulder"),
        smplnh_joints.index("right_shoulder")
    ],
    "hips":
    [smplnh_joints.index("left_hip"),
     smplnh_joints.index("right_hip")]
}

infos = {"mmm": mmm_joints_info, "smplnh": smplnh_joints_info}

smplh_indexes = {"mmm": smplh2mmm_indexes, "smplnh": smplh2smplnh_indexes}

root_joints = {
    "mmm": mmm_joints_info["root"],
    "mmmns": mmm_joints_info["root"],
    "smplmmm": mmm_joints_info["root"],
    "smplnh": smplnh_joints_info["root"],
    "smplh": smplh_joints.index("pelvis")
}


def get_root_idx(joinstype):
    return root_joints[joinstype]

coarse_bp = ['left arm', 'right arm', 'left leg', 'global orientation', 'right leg', 'torso']

less_coarse_bp = ['left arm',
                  'right arm',
                  'left leg',
                  'global orientation',
                  'right leg',
                  'torso',
                  'neck']

less_coarse_bp_v2 = ['left arm', 
                     'right arm',
                     'left leg',
                     'buttocks',
                     'right leg',
                     'torso',
                     'neck']

less_coarse_bp_v3 = ['left arm', 
                     'right arm',
                     'left leg',
                     'buttocks',
                     'waist',
                     'right leg',
                     'torso',
                     'neck']
