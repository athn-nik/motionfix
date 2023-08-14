from src.utils.nlp_consts import fix_spell
from src.data.tools.spatiotempo import temporal_compositions, spatial_compositions
from src.data.tools.spatiotempo import EXCLUDED_ACTIONS, EXCLUDED_ACTIONS_WO_TR


def extract_frame_labels_onlytext(babel_labels):
    seg_acts = []
    # if is_valid:
    if babel_labels['frame_ann'] is None:
        # 'transl' 'pose''betas'
        action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
        action_label = fix_spell(action_label)
        seg_acts.append(action_label)
    else:
        for seg_an in babel_labels['frame_ann']['labels']:
            action_label = fix_spell(seg_an['proc_label'])
            if action_label not in EXCLUDED_ACTIONS:
                seg_acts.append(action_label)

    return seg_acts


def extract_frame_labels(babel_labels, fps, seqlen, max_simultaneous):

    seg_ids = []
    seg_acts = []
    # is_valid = True
    # possible_frame_dtypes = ['seg', 'pairs', 'separate_pairs', 'spatial_pairs']
    # # if 'seq' in datatype and babel_labels['frame_ann'] is not None:
    # #     is_valid = False
    # if bool(set(datatype.split('+')) & set(possible_frame_dtypes)) \
    #     and babel_labels['frame_ann'] is None:
    #     is_valid = False

    possible_motions = {}
    # if is_valid:
    if babel_labels['frame_ann'] is None:
        # 'transl' 'pose''betas'
        action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
        possible_motions['seq'] = [(0, seqlen, fix_spell(action_label))]
    else:
        # Get segments
        # segments_dict = {k: {} for k in range(babel_labels['frame_ann']['labels'])}
        seg_list = []

        for seg_an in babel_labels['frame_ann']['labels']:
            action_label = fix_spell(seg_an['proc_label'])

            st_f = int(seg_an['start_t'] * fps)
            end_f = int(seg_an['end_t'] * fps)

            if end_f > seqlen:
                end_f = seqlen
            seg_ids.append((st_f, end_f))
            seg_acts.append(action_label)

            if action_label not in EXCLUDED_ACTIONS and end_f > st_f:
                seg_list.append((st_f, end_f, action_label))

        possible_motions['seg'] = seg_list
        spatial = spatial_compositions(seg_list,
                                       actions_up_to=max_simultaneous)
        possible_motions['spatial_pairs'] = spatial
        possible_motions['separate_pairs'] = temporal_compositions(
            seg_ids, seg_acts)

    return possible_motions
