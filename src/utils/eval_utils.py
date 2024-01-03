import torch

test_keyds= ["001895", "001893", "001892", "001888", "001848", "001881",
             "001827", "001845", "001858", "001812", "001798", "001755"]


keyids_for_testing = ["001892", "001881", "001800", "001790",
                      "001849", "001860", "001862",
                      "001641", "001696", "001676", "001690", "001692",
                      "001762", "001794", "001694", "001812", "001874",
                      "001883"]

def split_txt_into_multi_lines(input_str: str, line_length: int):
    words = input_str.split(' ')
    line_count = 0
    split_input = ''
    for word in words:
        line_count += 1
        line_count += len(word)
        if line_count > line_length:
            split_input += '\n'
            line_count = len(word) + 1
            split_input += word
            split_input += ' '
        else:
            split_input += word
            split_input += ' '
    
    return split_input

def regroup_metrics(metrics):
    from src.info.joints import smplh_joints
    pose_names = smplh_joints[1:23]
    dico = {key: val.numpy() for key, val in metrics.items()}

    if "APE_pose" in dico:
        APE_pose = dico.pop("APE_pose")
        for name, ape in zip(pose_names, APE_pose):
            dico[f"APE_pose_{name}"] = ape

    if "APE_joints" in dico:
        APE_joints = dico.pop("APE_joints")
        for name, ape in zip(smplh_joints, APE_joints):
            dico[f"APE_joints_{name}"] = ape

    if "AVE_pose" in dico:
        AVE_pose = dico.pop("AVE_pose")
        for name, ave in zip(pose_names, AVE_pose):
            dico[f"AVE_pose_{name}"] = ave

    if "AVE_joints" in dico:
        AVE_joints = dico.pop("AVE_joints")
        for name, ape in zip(smplh_joints, AVE_joints):
            dico[f"AVE_joints_{name}"] = ave

    return dico


def sanitize(dico):
    dico = {key: "{:.5f}".format(float(val)) for key, val in dico.items()}
    return dico

def out2blender(dicto):
    blender_dic = {}
    blender_dic['trans'] = dicto['body_transl']
    blender_dic['rots'] = torch.cat([dicto['body_orient'],
                                     dicto['body_pose']], dim=-1)
    return blender_dic