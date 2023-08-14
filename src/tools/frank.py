from src.info.joints import smpl2gpt, gpt_side, smpl_bps
from src.info.joints import smpl_bps_ids_list, smpl_bps_ids, smplh_joints
from src.info.joints import smpl_bps_ids

import string



def text_list_to_bp(text, gpt_labels_full_sent, return_original=False):
    if text in ['animal behavior series',
                'bird behavior series',
                'marine animal behavior series',
                'elephant behavior series', 
                'swim on air']:

        if return_original:
            return [1] * len(smpl_bps_ids), gpt_labels_full_sent[text][2]

        return [1] * len(smpl_bps_ids)
    else:
        original_cur_lbl = gpt_labels_full_sent[text][2]
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
        bp_list = [1 if x>1 else x for x in bp_list ]
        if bp_list == [0, 0 ,0, 0, 0, 0]:
            import ipdb; ipdb.set_trace()
        if return_original:
            return bp_list, original_cur_lbl
        return bp_list


def text_to_bp(text, gpt_labels_full_sent, return_original=False):
    original_cur_lbl = gpt_labels_full_sent[text]['GPT-response']
    cur_lbl = original_cur_lbl.translate(
        str.maketrans('', '', string.punctuation))
    # if 'whole body' in cur_lbl:
    #     bp_list = [1, 1, 1, 1, 1, 1]
    # else:
    cur_lbl = cur_lbl.lower().split(' ')
    bps = [
        bp for bp, wds_bp in smpl2gpt.items()
        if set(cur_lbl) & set(wds_bp)
    ]
    precise_bp = bps
    if [
            phr for phr in gpt_side['right']
            if phr in cur_lbl
    ]:
        precise_bp = [
            x for x in bps if not x.startswith('left')
        ]
    elif [
            phr for phr in gpt_side['left']
            if phr in cur_lbl
    ]:
        precise_bp = [
            x for x in bps if not x.startswith('right')
        ]
    # gpt_labels[f'{dtype}-{babel_id}-{index}'].append(precise_bp)

    bp_list = [0] * len(smpl_bps_ids)
    for bp_str in precise_bp:
        bp_list[smpl_bps_ids[bp_str]] += 1

    if return_original:
        return bp_list, original_cur_lbl
    return bp_list


def combine_motions(data1, data2, bp1, bp2, 
                    center=True, squeeze=False):
    import torch

    # minimum bp should be 2
    if sum(bp1) < sum(bp2):
        bp1, bp2 = bp2, bp1
        data1, data2 = data2, data1

    rots1, trans1 = data1.rots, data1.trans
    rots2, trans2 = data2.rots, data2.trans

    if squeeze:
        rots1, trans1 = torch.squeeze(rots1), torch.squeeze(trans1)
        rots2, trans2 = torch.squeeze(rots2), torch.squeeze(trans2)

    # STEP 1: same length with centering
    # common length
    length_1 = len(rots1)
    length_2 = len(rots2)

    # assumption 1
    # should be the same lenght
    length = min(length_1, length_2)

    if center:
        # force lenght constraint to be centered
        start_1 = (length_1 - length)//2
        rots1 = rots1[start_1:start_1+length]
        trans1 = trans1[start_1:start_1+length]

        start_2 = (length_2 - length)//2
        rots2 = rots2[start_2:start_2+length]
        trans2 = trans2[start_2:start_2+length]
    else:
        # trim length
        rots1 = rots1[:length]
        trans1 = trans1[:length]
        rots2 = rots2[:length]
        trans2 = trans2[:length]

    # assumption 2:
    # For composition, the two legs + global should be packed together
    left_leg_id = smpl_bps_ids_list.index("left leg")
    right_leg_id = smpl_bps_ids_list.index("right leg")
    global_id = smpl_bps_ids_list.index("global")

    if bp2[left_leg_id] or bp2[right_leg_id] or bp2[global_id]:
        bp1[left_leg_id] = 0
        bp1[right_leg_id] = 0
        bp1[global_id] = 0
        bp2[left_leg_id] = 1
        bp2[right_leg_id] = 1
        bp2[global_id] = 1
    else:
        bp1[left_leg_id] = 1
        bp1[right_leg_id] = 1
        bp1[global_id] = 1
        bp2[left_leg_id] = 0
        bp2[right_leg_id] = 0
        bp2[global_id] = 0
    # bp2 is minimum, will be added at the end (override)
    # add more to bp1

    # assumption 3:
    # binary selection of everything
    for i, x2 in enumerate(bp2):
        if x2 == 0:
            bp1[i] = 1

    body_parts_1 = [smpl_bps_ids_list[i] for i, x in enumerate(bp1) if x == 1]
    body_parts_1 = [y for x in body_parts_1 for y in smpl_bps[x]]

    body_parts_2 = [smpl_bps_ids_list[i] for i, x in enumerate(bp2) if x == 1]
    body_parts_2 = [y for x in body_parts_2 for y in smpl_bps[x]]

    # STEP 2: extract the body_parts
    joints_1 = [smplh_joints.index(x) for x in body_parts_1]
    joints_2 = [smplh_joints.index(x) for x in body_parts_2]

    frank_rots = torch.zeros_like(rots1)
    frank_rots[:, joints_1] = rots1[:, joints_1]
    frank_rots[:, joints_2] = rots2[:, joints_2]

    # assumption 3
    # gravity base hand crafted translation rule
    if "foot" in " ".join(body_parts_1):
        frank_trans = trans1
    else:
        frank_trans = trans2

    from src.transforms.smpl import RotTransDatastruct
    frank_data = RotTransDatastruct(rots=frank_rots, trans=frank_trans)
    return frank_data
