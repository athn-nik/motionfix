from src.utils.file_io import read_json, write_json
import os
import numpy as np
import joblib
import torch
from tqdm import tqdm

def get_file_list(directory, pattern):
    import glob
    files = glob.glob(f'{directory}/{pattern}')
    return files

def load_convert(path, lofs, gt_path, gt_dict, tgt2tgt=False):
    
    import ipdb; ipdb.set_trace()
    if path is not None:
        path_for_renders = f'fast-cluster/data/motion-editing-project/for_render'
    else:
        path_for_renders = 'fast-cluster/data/motion-editing-project/gt_renders'
    if 'sinc_synth' in path:
        amt = False
    else:
        amt = True

    os.makedirs(f'{path}/for_render', exist_ok=True)
    dict_to_save = {
                    'motion_a': [], 
                    'motion_b': [], 
                    'stamps_a': [], 
                    'stamps_b': []
                   }
    if path is not None:
        gt_dict_map = extract_gt_pairs(gt_dict, gt_path, 
                                       return_dict=True)

        for fname in tqdm(lofs):
            sample_gt = f'{gt_path}/{fname}.pth.tar'
            sample_npy_path = f'{path}/{fname}.npy'
            sample_pth_path = f'{path_for_renders}/{fname}.pth.tar'

            if os.path.exists(sample_npy_path):
                gen_motion = np.load(sample_npy_path,
                                    allow_pickle=True).item()['pose']
                gen_motion = torch.from_numpy(gen_motion)
                trans = gen_motion[..., :3]
                global_orient_6d = gen_motion[..., 3:9]
                body_pose_6d = gen_motion[..., 9:]
                cur_rots = torch.cat([global_orient_6d, body_pose_6d],
                                    dim=-1)
                sample = {'rots': cur_rots, 
                          'trans': trans}
                if not tgt2tgt:
                    dict_to_save['motion_b'].append(sample_pth_path)
                    dict_to_save['stamps_b'].append({'begin': 0,
                                                     'end': len(gen_motion)})

                    dict_to_save['motion_a'].append(gt_dict_map[fname]['motion_a'])
                    dict_to_save['stamps_a'].append({'begin': gt_dict_map[fname]['sta_a']['begin'],
                                                     'end': gt_dict_map[fname]['sta_a']['end']})
                else:
                    dict_to_save['motion_b'].append(sample_pth_path)
                    dict_to_save['stamps_b'].append({'begin': 0,
                                                     'end': len(gen_motion)})

                    dict_to_save['motion_a'].append(gt_dict_map[fname]['motion_b'])
                    dict_to_save['stamps_a'].append({'begin': gt_dict_map[fname]['sta_b']['begin'],
                                                     'end': gt_dict_map[fname]['sta_b']['end']})

                joblib.dump(sample, sample_pth_path)

    else:
        k_src, k_tgt, st_src, st_tgt = extract_gt_pairs(gt_dict, gt_path, 
                                                        return_dict=False)

        dict_to_save['motion_b'] = k_tgt
        dict_to_save['stamps_b'] = st_tgt
        dict_to_save['motion_a'] = k_src
        dict_to_save['stamps_a'] = st_src
    import ipdb; ipdb.set_trace()
    write_json(dict_to_save, f'{path_for_renders}/selected.json')
    return dict_to_save


def extract_gt_pairs(dict_to_loop, gt_path, return_dict=False):
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
            t_src = [int(s*30) for s in  v['timestamp_source']]
            t_tgt = [int(s*30) for s in  v['timestamp_target']]
            t_src_d = {'begin': t_src[0], 'end': t_src[1]}
            t_tgt_d = {'begin': t_tgt[0], 'end': t_tgt[1]}

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
        key_src_lst.append(f'{gt_path}/{key_src}.pth.tar')
        key_tgt_lst.append(f'{gt_path}/{key_tgt}.pth.tar')
        dictio[k] = {'motion_a': f'{gt_path}/{key_src}.pth.tar',
                     'motion_b': f'{gt_path}/{key_tgt}.pth.tar',
                     'sta_a': t_src_d,
                     'sta_b': t_tgt_d}

        # import ipdb; ipdb.set_trace()
    assert len(key_src_lst) == len(key_tgt_lst) == len(sta_src) == len(sta_tgt)
    if return_dict:
        return dictio
    return key_src_lst, key_tgt_lst, sta_src, sta_tgt

def main_loop(path, gt_path, gt_dict):
    from pathlib import Path
    if path is not None: # just the GT
        list_of_cands = get_file_list(path, '*candkeyids.json')
        cand_keys = []
        for candfile in list_of_cands:
            cand_keys.extend(read_json(candfile))
        cand_keys = list(set(cand_keys))
    else:
        cand_keys = None
    load_convert(path, cand_keys, gt_path, gt_dict)

    import ipdb;ipdb.set_trace()


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

    args = parser.parse_args()
    path = args.path
    dataset = args.ds
    if dataset == 'bodilex':
        gt_dict = read_json(path_amt_dict)
        gt_path = path_amt
    elif dataset == 'sinc_synth':
        gt_dict = read_json(path_sinc_dict)
        gt_path = path_sinc
    else:
        raise ValueError('Dataset not supported')
    main_loop(path, gt_path, gt_dict)
#  python visual_pkls.py --path experiments/clean-motionfix/bodilex/bs64_100ts_clip77/steps_1000_bodilex_noise_last/ld_txt-1.0_ld_mot-1.5 --ds bodilex

# 'all_candkeyids.json'
