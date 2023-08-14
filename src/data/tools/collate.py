from typing import List, Dict
from torch import Tensor
import torch


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_text_and_body_parts(lst_elements: List[Dict]) -> Dict:
    text_keys = [
        key for key in lst_elements[0] if isinstance(lst_elements[0][key], str)
    ]

    batch = {key: [x[key] for x in lst_elements] for key in text_keys}

    bp_gpt = collate_tensor_with_padding(
        [torch.tensor(x["bp-gpt"]) for x in lst_elements])

    batch["bp_gpt"] = bp_gpt
    return batch


def collate_datastruct_and_multi_text(lst_elements: List) -> Dict:
    # TODO if needed
    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate

    lst_element_pairs = [x["text"] for x in lst_elements]
    lst_elelents_others = []

    batch = {
        "single": {
            "datastruct":
            collate_datastruct([el["datastruct"] for el in lst_elements]),
            # Collate normally for the length
            "length": [x["length"] for x in lst_elements],
            # Collate the text
            "text": [x["text"] for x in lst_elements]
        },
        "multi": {}
    }

    #     }
    #     # train
    #     if "seperate" in batch:
    #         loss1 = self.compute_separete(batch["separate"])

    #         loss 2=

    #     if lst_elements[0] > 1:

    #    else:

    # batch = {
    #     # Collate with padding for the datastruct
    #     "datastruct": collate_datastruct([el["datastruct"] for el in lst_elements]),
    #     # Collate normally for the length
    #     "length": [x["length"] for x in lst_elements],
    #     # Collate the text
    #     "text": [x["text"] for x in lst_elements]
    #     }

    # # add keyid for example
    # otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    # for key in otherkeys:
    #     batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_datastruct_and_text(lst_elements: List) -> Dict:
    # collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
    batch = {# Collate with padding for the datastruct
             "datastruct": 
             collate_tensor_with_padding([el["datastruct"] for el in lst_elements]),
             # Collate normally for the length
            #  "datastruct_a": [x["datastruct_a"] for x in lst_elements],
            #  "datastruct_b": [x["datastruct_b"] for x in lst_elements],
             "length": [x["length"] for x in lst_elements],
             # Collate the text
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_length_and_text(lst_elements: List) -> Dict:

    batch = {
        "length_0": [x["length_0"] for x in lst_elements],
        "length_1": [x["length_1"] for x in lst_elements],
        "length_transition": [x["length_transition"] for x in lst_elements],
        "length_1_with_transition":
        [x["length_1_with_transition"] for x in lst_elements],
        "text_0": [x["text_0"] for x in lst_elements],
        "text_1": [x["text_1"] for x in lst_elements]
    }

    return batch


def collate_pairs_and_text(lst_elements: List, ) -> Dict:
    if 'features_0' not in lst_elements[0]:  # test set
        collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
        batch = {
            "datastruct":
            collate_datastruct([x["datastruct"] for x in lst_elements]),
            "length_0": [x["length_0"] for x in lst_elements],
            "length_1": [x["length_1"] for x in lst_elements],
            "length_transition":
            [x["length_transition"] for x in lst_elements],
            "length_1_with_transition":
            [x["length_1_with_transition"] for x in lst_elements],
            "text_0": [x["text_0"] for x in lst_elements],
            "text_1": [x["text_1"] for x in lst_elements]
        }

    else:
        batch = {
            "motion_feats_0":
            collate_tensor_with_padding(
                [el["features_0"] for el in lst_elements]),
            "motion_feats_1":
            collate_tensor_with_padding(
                [el["features_1"] for el in lst_elements]),
            "motion_feats_1_with_transition":
            collate_tensor_with_padding(
                [el["features_1_with_transition"] for el in lst_elements]),
            "length_0": [x["length_0"] for x in lst_elements],
            "length_1": [x["length_1"] for x in lst_elements],
            "length_transition":
            [x["length_transition"] for x in lst_elements],
            "length_1_with_transition":
            [x["length_1_with_transition"] for x in lst_elements],
            "text_0": [x["text_0"] for x in lst_elements],
            "text_1": [x["text_1"] for x in lst_elements]
        }
    return batch


def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {
        "length": [x["length"] for x in lst_elements],
        "text": [x["text"] for x in lst_elements]
    }

    # add keyid for example
    otherkeys = [
        x for x in lst_elements[0].keys()
        if x not in batch and x != "datastruct"
    ]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch
