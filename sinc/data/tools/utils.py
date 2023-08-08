import numpy as np
# from matplotlib import collections as mc
# from matplotlib.pyplot import cm
# import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

def separate_actions(pair: Tuple[Tuple]):

    if len(pair) == 3:
        if pair[0][1] < pair[2][0]:
            # a1 10, 15 t 14, 18 a2 17, 25
            # a1 10, 15 t 16, 16 a2 17, 25
            # transition only --> transition does not matter
            final_pair = [(pair[0][0], pair[1][0]),
                          (pair[1][0] + 1, pair[2][0] - 1),
                          (pair[2][0], pair[2][1])]
        else:
            # overlap + transition --> transition does not matter
            over = pair[2][0] - pair[0][1] 
            final_pair = [(pair[0][0], int(pair[0][1] + over/2)),
                          (int(pair[0][1] + over/2 + 1), pair[2][1])]
    else:
        # give based on small or long
        # p1_prop = (pair[0][1] - pair[0][0]) / (eoq - soq)
        # p2_prop = (pair[1][1] - pair[1][0]) / (eoq - soq)
        # over = pair[1][0] - pair[0][1]
        # final_pair = [(pair[0][0], int(p1_prop*over) + pair[0][1]),
        #               (int(p1_prop*over) + pair[0][1] + 1, pair[1][1])]

        # no transition at all
        over = pair[0][1] - pair[1][0] 
        final_pair = [(pair[0][0], int(pair[0][1] + over/2)),
                      (int(pair[0][1] + over/2 + 1), pair[1][1])]

    return final_pair

def timeline_overlaps(interval: Tuple, interval_list: List[Tuple]) -> Tuple[List[Tuple],
                                                               List[Tuple],
                                                               List[Tuple],
                                                               List[Tuple]]:
    '''
    Returns the intervals for which:
    (1) arr1 has overlap with
    (2) arr1 is a subset of
    (3) arr1 is a superset of
    '''
    l = interval[0]
    r = interval[1]
    inter_sub = []
    inter_super = []
    inter_before = []
    inter_after = []
    for s in interval_list:
        
        if (s[0] > l and s[0] > r) or (s[1] < l and s[1] < r):
            continue
        if s[0] <= l and s[1] >= r:
            inter_sub.append(s)
        if s[0] >= l and s[1] <= r:
            inter_super.append(s)
        if s[0] < l and s[1] < r and s[1] >= l:
            inter_before.append(s)
        if s[0] > l and s[0] <= r and s[1] > r:
            inter_after.append(s)

    return inter_before, inter_after

def segments_sorted(segs_fr: List[List], acts: List) -> Tuple[List[List], List]:

    assert len(segs_fr) == len(acts)
    if len(segs_fr) == 1: return segs_fr, acts
    L = [ (segs_fr[i],i) for i in range(len(segs_fr)) ]
    L.sort()
    sorted_segs_fr, permutation = zip(*L)
    sort_acts = [acts[i] for i in permutation]

    return  list(sorted_segs_fr), sort_acts


# def plot_timeline(segments, babel_id, outdir=get_original_cwd(), accel=None):

#     seg_ids = [(s_s, s_e) for s_s, s_e, _ in segments]
#     seg_acts = [f'{a}\n{s_s}|---|{s_e}' for s_s, s_e, a in segments]
#     seg_lns = [ [(x[0], i*0.01), (x[1], i*0.01)] for i, x in enumerate(seg_ids) ]
#     colorline = cm.rainbow(np.linsinc.0, 1, len(seg_acts)))
#     lc = mc.LineCollection(seg_lns, colors=colorline, linewidths=3,
#                             label=seg_acts)
#     fig, ax = plt.subplots()

#     ax.add_collection(lc)
#     fig.tight_layout()
#     ax.autoscale()
#     ax.margins(0.1)
#     # alternative for putting text there
#     # from matplotlib.lines import Line2D
#     # proxies = [ Line2D([0, 1], [0, 1], color=x) for x in colorline]
#     # ax.legend(proxies, seg_acts, fontsize='x-small', loc='upper left')
#     for i, a in enumerate(seg_acts):
#         plt.text((seg_ids[i][0]+seg_ids[i][1])/2, i*0.01 - 0.002, a,
#                  fontsize='x-small', ha='center')
#     if accel is not None:
#         plt.plot(accel)
#     plt.title(f'Babel Sequence ID\n{babel_id}')
#     plt.savefig(f'{outdir}/plot_{babel_id}.png')
#     plt.close()
