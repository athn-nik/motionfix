from src.utils.file_io import read_json
import string
import torch

smplh_joints = [
    # this is SMPL
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    # this is actually SMPLH
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
smpl_bps2ids = {
    'global': [0],
    'torso': [3, 6, 9, 12, 15],
    'left arm':[13, 16, 18, 20],
    'right arm':[14, 17, 19, 21],
    'left leg': [1, 4, 7, 10],
    'right leg':  [2, 5, 8, 11]
}

smpl_bps_ids = {
    'global': 0,
    'torso': 1,
    'left arm': 2,
    'right arm': 3,
    'left leg': 4,
    'right leg': 5
}
bp2ids = {
    bp_name: [smplh_joints.index(j) for j in jts_names]
    for bp_name, jts_names in smpl_bps.items()
    }

BODY_PART_DICT = read_json('deps/gpt/gpt3-labels-list.json')
BODY_PART_DICT_EDIT = read_json('deps/gpt/edit/gpt-labels_full.json')

def get_sinc_labels(list_of_texts):
    tot_list_bps = []
    for curtext in list_of_texts:
        bps_list = text_to_bp(curtext)
        tot_list_bps.append(bps_list)
    return tot_list_bps

def get_mask_from_bps(involved_jts_list, device, feat_dim=207):
    mask_all = torch.zeros((len(involved_jts_list),
                            feat_dim), dtype=torch.bool, device=device)
    for idx, sublist_jts in enumerate(involved_jts_list):
        for jt_idx in sublist_jts:

            if jt_idx == 0:
                # ROOT JOINT
                mask_all[idx, :15] = True
                mask_all[idx, 141: 144] = True
            else:
                mask_all[idx, 15 + (jt_idx-1)*6 :15 + (jt_idx-1)*6 + 6] = True
                mask_all[idx, 141 + (jt_idx-1)*3 :141 + (jt_idx-1)*3 + 3] = True
    return mask_all

def get_mask_from_texts(list_of_texts):
    jts_from_txt = get_jts_from_bps(get_sinc_labels(list_of_texts))
    return jts_from_txt

def text_to_bp(text, return_original=False):
    if text in ['animal behavior series',
                'bird behavior series',
                'marine animal behavior series',
                'elephant behavior series', 
                'swim on air']:

        if return_original:
            if text in BODY_PART_DICT:
                return [1] * len(smpl_bps_ids), BODY_PART_DICT[text][2]
            else:
                return [1] * len(smpl_bps_ids), BODY_PART_DICT_EDIT[text][2]
        return [1] * len(smpl_bps_ids)
    else:
        if text in BODY_PART_DICT:
            original_cur_lbl = BODY_PART_DICT[text][2]
        else:
            original_cur_lbl = BODY_PART_DICT_EDIT[text]
        cur_lbl = original_cur_lbl.translate(str.maketrans('', '',
                                                        string.punctuation))
        cur_lbl = cur_lbl.lower().replace('answer', '')
        precise_bp = cur_lbl.strip().split('\n')
        if 'right side of face' in precise_bp:
            precise_bp.remove('right side of face')
        if 'right eye' in precise_bp:
            precise_bp.remove('right eye')
        if 'wrist' in precise_bp:
            precise_bp.remove('wrist')

        if 'hand' in precise_bp: precise_bp.remove('hand')
        # if set(precise_bp) & set(['hand', 'left hand', 'right hand']):
        #     if 'right arm' in precise_bp and 'right hand' in precise_bp:
        #         precise_bp.remove('right hand')
        #     elif 'left arm' in precise_bp and 'left hand' in precise_bp:
        #         precise_bp.remove('left hand')
        #     else:
        #         if 'hand' in precise_bp:
        #             precise_bp.remove('hand')
        #         elif 'rig'
        #         precise_bp.append('right arm')
        #         precise_bp.append('left arm')

        if 'shoulders' in precise_bp:
            if 'right arm' in precise_bp or 'left arm' in precise_bp:
                precise_bp.remove('shoulders')
            else:
                precise_bp.append('right arm')
                precise_bp.append('left arm')
                precise_bp.remove('shoulders')
 
        if 'right shoulder' in precise_bp:
            precise_bp.append('right arm')
            precise_bp.remove('right shoulder')
 
        if 'right shoulder' in precise_bp:
            precise_bp.append('right arm')
        if 'left shoulder' in precise_bp:
            precise_bp.append('left arm')
            
        bp_list = [0] * len(smpl_bps_ids)
        
        for bp_str in precise_bp:
            if bp_str.strip()  == 'buttocks' or bp_str.strip() =='waist':
                bp_final= 'global'
            elif bp_str.strip() == 'neck':
                bp_final = 'torso'
            else:
                bp_final = str(bp_str.strip())

            if bp_final in smpl_bps_ids:
                x=1
            else:
                import ipdb;ipdb.set_trace()
            bp_list[smpl_bps_ids[bp_final]] += 1
        bp_list = [1 if x>1 else x for x in bp_list ]

        assert bp_list != [0, 0 ,0, 0, 0, 0]
        if return_original:
            return bp_list, original_cur_lbl
        return bp_list


def get_jts_from_bps(batch_of_bplists):
    bp_batch_jts_idxs = []
    for bp_list in batch_of_bplists:
        cur_bp_list = []
        if bp_list[0]:
            cur_bp_list += smpl_bps2ids['global']
        if bp_list[1]:
            cur_bp_list += smpl_bps2ids['torso']
        if bp_list[2]:
            cur_bp_list += smpl_bps2ids['left arm']
        if bp_list[3]:
            cur_bp_list += smpl_bps2ids['right arm']
        if bp_list[4]:
            cur_bp_list += smpl_bps2ids['left leg']
        if bp_list[5]:
            cur_bp_list += smpl_bps2ids['right leg']
        bp_batch_jts_idxs.append(cur_bp_list)

    return bp_batch_jts_idxs
