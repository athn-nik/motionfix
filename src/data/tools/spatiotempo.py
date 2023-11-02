import itertools
from src.data.tools.utils import segments_sorted, timeline_overlaps
from typing import Dict, List, Optional, Tuple
from operator import itemgetter
from src.data.tools.utils import separate_actions

EXCLUDED_ACTIONS = ['t-pose', 'a-pose', 'a pose','t pose', 
                    'tpose', 'apose', 'transition']
EXCLUDED_ACTIONS_WO_TR = ['t-pose', 'a-pose', 'a pose','t pose', 'tpose', 'apose']


def temporal_compositions(seg_ids, seg_acts):
    import itertools
    seg_ids, seg_acts = segments_sorted(seg_ids, seg_acts)

    # remove a/t pose for pair calculation
    seg_acts_for_pairs = [a for a in seg_acts if a not in EXCLUDED_ACTIONS_WO_TR ]
    idx_to_keep = [i for i, a in enumerate(seg_acts) if a not in EXCLUDED_ACTIONS_WO_TR ]
    seg_ids_for_pairs = [s for i, s in enumerate(seg_ids) if i in idx_to_keep]
    assert len(seg_acts_for_pairs) == len(seg_ids_for_pairs)

    seg2act = dict(zip(seg_ids_for_pairs, seg_acts_for_pairs))
    # plot_timeline(seg_ids, seg_acts, babel_key)

    overlaps_for_each_seg = {}
    for segment in seg_ids_for_pairs:
        # remove the segment of interest
        seg_ids_wo_seg = [x for x in seg_ids_for_pairs if x != segment]
        # calculate the before and after overlaps for the segment of interest
        ov_bef, ov_aft = timeline_overlaps(segment, seg_ids_wo_seg)

        overlaps_for_each_seg[segment] = {}
        overlaps_for_each_seg[segment]['before'] = ov_bef
        overlaps_for_each_seg[segment]['after'] = ov_aft

    pairs_s = []
    pairs_a = []
    for seg_, ov_seg in overlaps_for_each_seg.items():
        cur_act_pairs = []
        cur_seg_pairs = []
        cur_seg_pairs_bef = []
        cur_seg_pairs_af = []
        if seg2act[seg_] == 'transition':
            # if transition is not the start
            if not seg_[0] == 0:
                if ov_seg['before'] and ov_seg['after']:
                    cur_seg_pairs = list(itertools.product(ov_seg['before'], ov_seg['after']))
                    cur_act_pairs = [(seg2act[x], seg2act[y]) for x, y in cur_seg_pairs]
                    
                    cur_seg_pairs = [tuple(sorted(p, key=lambda item: item[0])) for p in cur_seg_pairs]
                    cur_seg_pairs = [(a, seg_, b) for a,b in cur_seg_pairs]

                    pairs_s.append(cur_seg_pairs)
                    pairs_a.append(cur_act_pairs)

        else:
            ov_seg['before'] = [x for x in ov_seg['before'] if seg2act[x] != 'transition']
            ov_seg['after'] = [x for x in ov_seg['after'] if seg2act[x] != 'transition']
            if ov_seg['before']:
                cur_seg_pairs_bef = list(itertools.product(ov_seg['before'], [seg_]))
            if ov_seg['after']:
                cur_seg_pairs_af = list(itertools.product([seg_], ov_seg['after']))

            if ov_seg['after'] and ov_seg['before']:
                cur_seg_pairs = cur_seg_pairs_bef + cur_seg_pairs_af
            elif ov_seg['after']:
                cur_seg_pairs = cur_seg_pairs_af
            elif ov_seg['before']:
                cur_seg_pairs = cur_seg_pairs_bef
            else:
                continue

            cur_seg_pairs = [tuple(sorted(p, key=lambda item: item[0])) for p in cur_seg_pairs]
            cur_act_pairs = [(seg2act[x], seg2act[y]) for x, y in cur_seg_pairs]

            # separate_pairs
            # [((),())]
            # list of tuples for everything except separate_pairs
            pairs_s.append(cur_seg_pairs)
            pairs_a.append(cur_act_pairs)

    # flatten list of lists
    pairs_s = list(itertools.chain(*pairs_s))
    pairs_a = list(itertools.chain(*pairs_a))

    # remove duplicates
    from more_itertools import unique_everseen

    tmp = zip(pairs_s, pairs_a)
    uniq_tmp = unique_everseen(tmp, key=itemgetter(0))
    segment_pairs = []
    action_pairs = []
    for seg, a in list(uniq_tmp):
        segment_pairs.append(seg)
        action_pairs.append(a)

    assert len(segment_pairs) == len(action_pairs)

    motion_pairs = []
    final_pairs = []
    for seg in segment_pairs:
        fpair = separate_actions(seg)
        final_pairs.append(tuple(fpair))
    for idx, durs in enumerate(final_pairs):
        motion_pairs.append((durs, action_pairs[idx]))
    return motion_pairs


    
def spatial_compositions(segments: List[Tuple], actions_up_to=2) -> List[Tuple]:
    # input = list of (start, stop, symbol) tuples
    points = [] # list of (offset, plus/minus, symbol) tuples
    actions = [a for _, _, a in segments]
    seen_acts = set()
    counter = {}
    if len(set(actions)) != len(segments):
        for idx, act in enumerate(actions):
            if act in seen_acts:
                actions[idx] += f'---occ{counter[act]+1}'
                counter[act] += 1
            else:
                seen_acts.add(act)
                counter[act] = 1

    for idx, (start, stop, action) in enumerate(segments):
        points.append((start, '<s>', actions[idx]))
        points.append((stop, '<e>', actions[idx]))
    points.sort()

    ranges = [] # output list of (start, stop, symbol_set) tuples
    current_set = set()
    last_start = None
    for offset, end_or_start, action in points:
        if end_or_start == '<s>':
            if last_start is not None:
                ranges.append((last_start, offset, current_set.copy()))
            current_set.add(action)
            last_start = offset
        elif end_or_start == '<e>':

            # Getting a minus without a last_start is possible i.e. gap between actions
            if len(current_set) == 1 and action == list(current_set)[0]:
                last_start = None
                current_set = set()
            else:
                # if len(current_set) > 0: # multiple same actions
                ranges.append((last_start, offset, current_set.copy()))
                current_set.remove(action)
                last_start = offset
    # Finish off
    if last_start is not None:
        ranges.append((last_start, offset, current_set.copy()))

    spatial_segments = [[s, e, tuple(action_set)] for s, e, action_set in ranges if s < e and len(action_set) > 1]
    import re
    for idx, (_, _, action_set) in enumerate(spatial_segments):
        action_set_proc = [ac.split("---")[0] for ac in action_set]
        spatial_segments[idx][-1] = tuple(action_set_proc) 
    spatial_segments = [sp for sp in spatial_segments if not (len(sp[2]) == 2 and 'transition' in sp[2])]
    spatial_segments = [sp for sp in spatial_segments if len(sp[2]) <= actions_up_to ]

    return spatial_segments
