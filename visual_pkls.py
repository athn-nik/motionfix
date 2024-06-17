from src.utils.file_io import read_json, write_json
import os
import numpy as np
import joblib
import torch
from tqdm import tqdm
import src.launch.prepare  # noqa

def get_file_list(directory, pattern):
    import glob
    files = glob.glob(f'{directory}/{pattern}')
    return files


def find_gt(search_value):
    """
    Function to find the key corresponding to a given value in a dictionary where values are lists.
    
    Parameters:
        search_dict (dict): The dictionary to search through.
        search_value (str): The value to search for in the dictionary's value lists.
    
    Returns:
        str: The key corresponding to the found value, or None if the value is not found.
    """
    ss_seq2key = '/fast/nathanasiou/logs/blender_motionfix/info/sinc_synth_seq2keys.json'
    search_dict = read_json(ss_seq2key)
    # Iterate over each key and list of values in the dictionary
    for key, values in search_dict.items():
        # Check if the search_value is in the current list of values
        if search_value in values:
            return key  # Return the key if the value is found
    return None  # Return None if the value is not found in any list



def load_convert(path, lofs, gt_path, gt_dict, tgt2tgt=False):
    
    # import ipdb; ipdb.set_trace()
    if path is not None:
        path_for_renders = f'fast-cluster/logs/blender_motionfix/results_render/pkls'
        pts_gen = '/'.join(path.split('/')[2:])
        path_for_renders = f'{path_for_renders}/{pts_gen}'
        os.makedirs(path_for_renders, exist_ok=True)
    else:
        path_for_renders = 'fast-cluster/logs/blender_motionfix/gt_renders'
    if 'sinc_synth' in gt_path:
        amt = False
        # path_for_renders = 'fast-cluster/logs/blender_motionfix/gt_renders/sinc_synth'
    else:
        amt = True
        # path_for_renders = 'fast-cluster/logs/blender_motionfix/gt_renders/bodilex'

    # os.makedirs(f'{path}/for_render', exist_ok=True)
    dict_to_save = {
                    'motion_a': [], 
                    'motion_b': [], 
                    'stamps_a': [], 
                    'stamps_b': []
                   }
    from src.tools.transforms3d import transform_body_pose
    if path is not None:
        gt_dict_map = extract_gt_pairs(gt_dict, gt_path, 
                                       return_dict=True)

        for fname in tqdm(lofs):
            sample_gt = f'{gt_path}/{fname}.pth.tar'
            sample_npy_path = f'{path}/{fname}.npy'
            sample_pth_path = f'{path_for_renders}/{fname}.pth.tar'
            # import ipdb;ipdb.set_trace() 
            if os.path.exists(sample_npy_path):
                gen_motion = np.load(sample_npy_path,
                                    allow_pickle=True).item()['pose']
                gen_motion = torch.from_numpy(gen_motion)
                trans = gen_motion[..., :3]
                body_pose = transform_body_pose(gen_motion[..., 3:], f"6d->aa")
                global_orient_6d = body_pose[..., :3]
                body_pose_6d = body_pose[..., 3:]
                cur_rots = torch.cat([global_orient_6d, body_pose_6d],
                                    dim=-1)
                sample = {'rots': cur_rots, 
                          'trans': trans}
                if not tgt2tgt:
                    dict_to_save['motion_b'].append(sample_pth_path.replace('.pth.tar',''))
                    dict_to_save['stamps_b'].append({'begin': 0,
                                                     'end': len(gen_motion)})
                    
                    dict_to_save['motion_a'].append(gt_dict_map[fname]['motion_a'].replace('.pth.tar',''))
                    dict_to_save['stamps_a'].append({'begin': gt_dict_map[fname]['sta_a']['begin'],
                                                     'end': gt_dict_map[fname]['sta_a']['end']})
                else:
                    dict_to_save['motion_b'].append(sample_pth_path.replace('.pth.tar',''))
                    dict_to_save['stamps_b'].append({'begin': 0,
                                                     'end': len(gen_motion)})

                    dict_to_save['motion_a'].append(gt_dict_map[fname]['motion_b'].replace('.pth.tar',''))
                    dict_to_save['stamps_a'].append({'begin': gt_dict_map[fname]['sta_b']['begin'],
                                                     'end': gt_dict_map[fname]['sta_b']['end']})
                # import ipdb;ipdb.set_trace()
                joblib.dump(sample, sample_pth_path)

    else:
        k_src, k_tgt, st_src, st_tgt = extract_gt_pairs(gt_dict, gt_path, 
                                                        return_dict=False)

        dict_to_save['motion_b'] = k_tgt
        dict_to_save['stamps_b'] = st_tgt
        dict_to_save['motion_a'] = k_src
        dict_to_save['stamps_a'] = st_src
    # import ipdb; ipdb.set_trace()
    if tgt2tgt:
        extra_str = 't2t'
    else:
        extra_str = 's2t'
    print(f'The path that you can grab a json from and run the renderings is in\n{path_for_renders}/selected_{extra_str}.json')
    #import ipdb; ipdb.set_trace()
    write_json(dict_to_save, f'{path_for_renders}/selected_{extra_str}.json')
    return dict_to_save


def extract_gt_pairs(dict_to_loop, gt_path, return_dict=False):
    import os

    IS_LOCAL_DEBUG = src.launch.prepare.get_local_debug()
    if 'sinc_synth' in gt_path:
        if IS_LOCAL_DEBUG:
            path_to_d = 'fast-cluster/logs/blender_motionfix/info/sinc_synth.json'
        else:
            path_to_d = '/fast/nathanasiou/logs/blender_motionfix/info/sinc_synth.json'
        
        if os.path.exists(path_to_d):
            print("GT Data file already exists!")
            dictio = read_json(path_to_d)
            if return_dict:
                return dictio
    else:
        if IS_LOCAL_DEBUG:
            path_to_d = 'fast-cluster/logs/blender_motionfix/info/bodilex.json'
        else:
            path_to_d = '/fast/nathanasiou/logs/blender_motionfix/info/bodilex.json'
        
        if os.path.exists(path_to_d):
            print("GT Data file already exists!")
            dictio = read_json(path_to_d)
            if return_dict:
                return dictio

    sta_src = []
    sta_tgt = []
    key_src_lst = []
    key_tgt_lst = []
    if 'sinc_synth' in gt_path:
        amt = False
    else:
        amt = True
    dictio = {}
    for k, v in tqdm(dict_to_loop.items()):
        if amt:
            key_src = v['motion_source']
            key_tgt = v['motion_target']
            
            kdur_src = v['motion_a'].split('/')[-1].replace('.mp4','')
            dur_src = kdur_src.split('_')[-2:]

            kdur_tgt = v['motion_b'].split('/')[-1].replace('.mp4','')
            dur_tgt = kdur_tgt.split('_')[-2:]
            assert int(dur_src[0]) < int(dur_src[1])
            assert int(dur_tgt[0]) < int(dur_tgt[1])
            t_src_d = {'begin': int(dur_src[0]), 'end': int(dur_src[1])}
            t_tgt_d = {'begin': int(dur_tgt[0]), 'end': int(dur_tgt[1])}
            
            # import ipdb;ipdb.set_trace()
        else:
            key_src = v['source_babel_key']
            key_tgt = v['target_babel_key']

            vkey_src = v['motion_a'].replace('https://motion-editing.s3.eu-central-1.amazonaws.com/sinc_synth/rendered_source/', '').replace('.mp4', '')
            vkey_tgt = v['motion_b'].replace('https://motion-editing.s3.eu-central-1.amazonaws.com/sinc_synth/rendered_target/', '').replace('.mp4', '')

            dur_src = vkey_src.replace(key_src, '').split('_')[1:]
            dur_tgt = vkey_tgt.replace(key_tgt, '').split('_')[1:]
            dur_src = [int(d) for d in dur_src]
            dur_tgt = [int(d) for d in dur_tgt]
            assert dur_tgt == dur_src
            
            t_src_d = {'begin': dur_src[0], 'end': dur_src[1]}
            t_tgt_d = {'begin': dur_tgt[0], 'end': dur_tgt[1]}

        sta_src.append(t_src_d)
        sta_tgt.append(t_tgt_d)
        assert os.path.exists(f'{gt_path}/{key_src}.pth.tar')
        assert os.path.exists(f'{gt_path}/{key_tgt}.pth.tar')
        key_src_lst.append(f'{gt_path}/{key_src}')
        key_tgt_lst.append(f'{gt_path}/{key_tgt}')
        dictio[k] = {'motion_a': f'{gt_path}/{key_src}',
                     'motion_b': f'{gt_path}/{key_tgt}',
                     'sta_a': t_src_d,
                     'sta_b': t_tgt_d}

    assert len(key_src_lst) == len(key_tgt_lst) == len(sta_src) == len(sta_tgt)
    if 'sinc_synth' in gt_path:
        path_to_d = '/fast/nathanasiou/logs/blender_motionfix/info/sinc_synth.json'
        write_json(dictio, path_to_d)
    else:
        path_to_d = '/fast/nathanasiou/logs/blender_motionfix/info/bodilex.json'
        write_json(dictio, path_to_d)
    import ipdb; ipdb.set_trace()
    if return_dict:
        return dictio
    return key_src_lst, key_tgt_lst, sta_src, sta_tgt

def main_loop(path, gt_path, gt_dict, mode, set_to_pick):
    from pathlib import Path
    if path is not None: # just the GT
        if set_to_pick == 'best':
            list_of_cands = get_file_list(path, '*candkeyids.json')
            cand_keys = []
            for candfile in list_of_cands:
                cand_keys.extend(read_json(candfile))
        else:
            cand_keys = []
            splits = read_json('data/bodilex/splits.json')
            cand_keys = splits['train'] +splits['val']
        # import ipdb;ipdb.set_trace()
        cand_keys = list(set(cand_keys))
    else:
        cand_keys = None
    assert mode in ['t2t', 's2t']
    tgt2tgt = False if mode =='s2t' else True
    load_convert(path, cand_keys, gt_path, gt_dict, tgt2tgt)
    
    # import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    import argparse
    path_amt = 'fast-cluster/data/bodilex/hml3d_v5_pos_canonica_fixed/perfile'
    path_amt_dict = 'fast-cluster/data/motion-editing-project/bodilex/amt_motionfix_latest.json'
    path_sinc = 'fast-cluster/data/motion-editing-project/sinc_synth/pkls'
    path_sinc_dict = 'fast-cluster/data/motion-editing-project/sinc_synth/for_website_v4.json'

    # run_dir=outputs/tmr_humanml3d_amass_feats
    # samples_path=experiments/clean-motionfix/bodilex/bs64_100ts_clip77/steps_1000_bodilex_noise_last/ld_txt-1.0_ld_mot-1.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=False, type=str,
                        help='dataset', default=None)
    parser.add_argument('--ds', required=True, type=str,
                        help='dataset')
    parser.add_argument('--mode', required=True, type=str,
                        help='dataset')
    parser.add_argument('--set', required=False, default='best', type=str,
                        help='dataset')
    args = parser.parse_args()
    path = args.path
    mode = args.mode
    dataset = args.ds
    if dataset == 'bodilex':
        gt_dict = read_json(path_amt_dict)
        gt_path = path_amt
    elif dataset == 'sinc_synth':
        gt_dict = read_json(path_sinc_dict)
        gt_path = path_sinc
    else:
        raise ValueError('Dataset not supported')
    main_loop(path, gt_path, gt_dict, mode, args.set)
#  python visual_pkls.py --path experiments/clean-motionfix/bodilex/bs64_100ts_clip77/steps_1000_bodilex_noise_last/ld_txt-1.0_ld_mot-1.5 --ds bodilex

# 'all_candkeyids.json'
