from typing import Optional
import logging
import joblib

from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

from sinc.utils.file_io import read_json
from .base import BASEDataModule
from sinc.data.tools.extract_pairs import extract_frame_labels, extract_frame_labels_onlytext
from sinc.info.joints import smpl2gpt, gpt_side
import string

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


def get_babel_keys(path: str):
    filepath = Path(path) / f'../babel_v2.1/id2fname/amass-path2babel.json'
    amass2babel = read_json(filepath)
    return amass2babel


def load_gpt_labels(path: str):
    gpt_labels = read_json(path)
    return gpt_labels


class GPT3ProxyDataModule(BASEDataModule):
    def __init__(self,
                 data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         datatype="text")
        self.save_hyperparameters(logger=False)
        self.Dataset = GPT3Proxy
        sample_overrides = {"split": "train", "progress_bar": False}

        self._sample_set = self.get_sample_set(overrides=sample_overrides)


class GPT3Proxy(Dataset):
    dataname = "GPT3Proxy"

    def __init__(self,
                 datapath: str,
                 split: str = "train",
                 progress_bar: bool = True,
                 **kwargs):

        self.split = split
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()
        gpt_labels_full_sent = load_gpt_labels(kwargs['gpt_path'])
        self.babel_annots = read_json(
            Path(datapath) / f'../babel_v2.1/{split}.json')

        texts_data = {}
        gpt_sents = {}
        gpt_bps = {}

        if progress_bar:
            enumerator = tqdm(self.babel_annots.items(),
                              f"Loading BABEL {split}")
        else:
            enumerator = self.babel_annots.items()

        num_bad = 0
        num_good = 0
        for babel_id, labels in enumerator:
            # import ipdb
            # ipdb.set_trace()
            seg_acts = extract_frame_labels_onlytext(labels)

            for index, action in enumerate(seg_acts):
                if action == "":
                    continue

                keyid = f'{babel_id}-{index}'

                if action not in gpt_labels_full_sent:
                    num_bad += 1
                    continue

                num_good += 1
                texts_data[keyid] = action

                cur_lbl = gpt_labels_full_sent[action]['GPT-response']
                gpt_sents[keyid] = cur_lbl

                cur_lbl = cur_lbl.translate(
                    str.maketrans('', '', string.punctuation))

                if 'whole body' in cur_lbl:
                    bp_list = [1, 1, 1, 1, 1, 1]
                else:
                    cur_lbl = cur_lbl.lower().split(' ')
                    bps = [
                        bp for bp, wds_bp in smpl2gpt.items()
                        if set(cur_lbl) & set(wds_bp)
                    ]
                    precise_bp = bps
                    if [phr for phr in gpt_side['right'] if phr in cur_lbl]:
                        precise_bp = [
                            x for x in bps if not x.startswith('left')
                        ]
                    elif [phr for phr in gpt_side['left'] if phr in cur_lbl]:
                        precise_bp = [
                            x for x in bps if not x.startswith('right')
                        ]
                    from sinc.info.joints import smpl_bps_ids
                    bp_list = [0] * len(smpl_bps_ids)
                    for bp_str in precise_bp:
                        bp_list[smpl_bps_ids[bp_str]] += 1

                gpt_bps[keyid] = bp_list

        logger.info(
            f"The number of bad action are: {num_bad}. The ratio is {num_bad/(num_bad+num_good)}"
        )
        self.texts_data = texts_data
        self.keyids = list(self.texts_data.keys())
        self._split_index = self.keyids
        self.gpt_sents = gpt_sents
        self.gpt_bps = gpt_bps

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences

    def load_keyid(self, keyid):
        text = self._load_text(keyid)
        element = {
            'text': text,
            'keyid': keyid,
            'bp-gpt': self.gpt_bps[keyid],
            'sent-gpt': self.gpt_sents[keyid]
        }
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"
