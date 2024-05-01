import os
from openai import OpenAI
import logging
from tqdm import tqdm
from src.utils.file_io import read_json, write_json
import time
from rich import print 

logger = logging.getLogger(__name__)


# body_parts = ['left arm', 'right arm', 'left leg', 'global orientation',
# 'right leg', 'torso', 'left hand', 'right hand', 'left ankle', 'right ankle', 'left foot',
# 'right foot', 'head', 'neck', 'right shoulder', 'left shoulder', 'pelvis', 'spine']

# # fine_bp = list(body_parts)
# coarse_bp = ['left arm', 'right arm', 'left leg', 'global orientation', 'right leg', 'torso']
# coarse_bp_v2 = ['left arm', 'right arm', 'left leg', 'global orientation',
#                 'right leg', 'torso', 'head', 'neck', 'pelvis']

# 'What parts of the body are moving when someone is doing the action:'


def gpt_extract():

    from src.utils.file_io import write_json
    from gpt_parts.prompts import edit_prompts_to_gpt
    unique_texts = ["don't move upper body"]
    ds_amt = read_json('data/bodilex/amt_motionfix_latest.json')
    unique_texts = []
    for k, v in ds_amt.items():
        unique_texts.append(v['annotation'])
    pathout='deps/gpt/edit/gpt-labels_full.json'
    already_existing_dict = read_json(pathout)
    already_existing = list(already_existing_dict.keys())
    unique_texts = list(set(unique_texts))

    unique_texts = [item for item in unique_texts if item not in already_existing]
    responses = {}
    responses_full = dict(already_existing_dict)
    write_every = 50
    elems = 0
    cur_model = 'gpt-3.5-turbo'
    n_to_keep = len(unique_texts)
    from openai import OpenAI
    import ipdb; ipdb.set_trace() 
    client = OpenAI()
    if unique_texts:
        batch_id = 0
        for action_text in tqdm(unique_texts[:n_to_keep]):
            prompts = edit_prompts_to_gpt.replace('[EDIT TEXT]', action_text)
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompts}])

            response_compl = response.choices[0].message.content.strip()
            responses[action_text] = response_compl
            responses_full[action_text] = response_compl
            elems += 1
            if elems % write_every == 0:
                batch_id = elems // write_every
                write_json(responses, f'deps/gpt/edit/gpt-labels_batch{batch_id}.json')
                responses = {}
                time.sleep(30)
            # if elems == 100:
            #    time.sleep(120)
        # final batch
        batch_id += 1
        write_json(responses, f'deps/gpt/edit/gpt-labels_batch{batch_id}.json')
        # write html/tex to file
        import ipdb; ipdb.set_trace()
        write_json(responses_full, pathout)
    else:
        print(f'All texts exists in {pathout}')
    # p2 = '/home/nathanasiou/Desktop/gpt3-labels.json'
    # write_json(responses_full, p2)


if __name__ == '__main__':
    gpt_extract()
