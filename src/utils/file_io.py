import json
from PIL import Image
# import cv2
import subprocess
import numpy as np
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
from collections import Counter
import glob
import hydra
import yaml
from pathlib import Path
import codecs as cs
def read_text_lines(path, split):
    split_file = os.path.join(path, split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

def get_metric_paths(sample_path: Path, set: str, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    if set == 'pairs':
        metric_str = "babel_metrics_sp" 
    else:
        metric_str = f"babel_metrics_{set}"

    if onesample:
        file_path = f"{fact_str}{metric_str}_{split}{extra_str}"
        save_path = sample_path / file_path
        return save_path
    else:
        file_path = f"{fact_str}{metric_str}_{split}_multi"
        avg_path = sample_path / (file_path + "_avg")
        best_path = sample_path / (file_path + "_best")
        return avg_path, best_path


def hack_path(path, keyword='deps'):
    # workaround for cluster local/sync
    rel_p = path.split('/')
    ########################################################
    rel_p = rel_p[rel_p.index(keyword):]
    rel_p = '/'.join(rel_p)
    path = hydra.utils.get_original_cwd() + '/' + rel_p
    return path

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def get_samples_folder(path, ckpt, * , jointstype):
    if jointstype == "vertices":
        raise ValueError("No evaluation for vertices, sample the joints instead.")

    output_dir = Path(hydra.utils.to_absolute_path(path))
    candidates = [x for x in os.listdir(output_dir) if "samples" in x]
    if not candidates:
        raise ValueError("There is no samples for this model.")

    samples_path = output_dir / f"samples" / f'checkpoint-{ckpt}'

    return samples_path, jointstype

def to_vtt(frames, fps, acts, fname):
    import datetime
    # str(datetime.timedelta(seconds=666))
    # duration = max(max(frames)) / fps

    with open(fname, 'w') as f:
        f.write('WEBVTT\n\n')
        for f_pair, a in zip(frames, acts):
            f_start, f_end = f_pair

            def format_time(input_secs):
                hours , remainder = divmod(input_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                mins = str(int(minutes)).zfill(2)
                secs = str(round(seconds, 3)).zfill(6)
                hours = str(hours).zfill(2)
                return hours,mins, secs

            hs, ms, ss = format_time(f_start/fps)
            he, me, se = format_time(f_end/fps)

            f.write(f'{hs}:{ms}:{ss} --> {he}:{me}:{se}\n')
            f.write(f'{a}\n\n')

    return fname

def to_srt(frames, fps, acts, fname):
    import datetime
    # str(datetime.timedelta(seconds=666))
    # duration = max(max(frames)) / fps
    ln = 1
    with open(fname, 'w') as f:

        for f_pair, a in zip(frames, acts):
            f_start, f_end = f_pair
            f.write(f'{ln}\n')
            ln += 1
            def format_time(input_secs):
                hours , remainder = divmod(input_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                mins = str(int(minutes)).zfill(2)
                secs = str(round(seconds, 3)).zfill(6).replace('.', ',')
                hours = str(hours).zfill(2)
                return hours, mins, secs

            hs, ms, ss = format_time(f_start/fps)
            he, me, se = format_time(f_end/fps)

            f.write(f'{hs}:{ms}:{ss} --> {he}:{me}:{se}\n')
            f.write(f'{a}\n')
    # ffmpeg -i finalOutput.mp4 -vf subtitles=subs.srt out.mp4
    return fname

def read_json(p):
    with open(p, 'r') as fp:
        json_contents = json.load(fp)
    return json_contents

def write_json(data, p):
    with open(p, 'w') as fp:
        json.dump(data, fp, indent=2)

# Load npys
def loadnpys(path: str):
    import glob
    dict_of_npys = {}
    import numpy as np
    for p in glob.glob(f'{path}/*.npy'):
        data_sample = np.load(p, allow_pickle=True).item()
        fname = p.split('/')[-1]
        keyid = fname.replace('.npy', '')
        dict_of_npys[keyid] = (data_sample['text'], data_sample['lengths'])
    return dict_of_npys

class Video:
    # Credits to Lucas Ventura
    def __init__(self, frame_path: str, fps: float = 12.5):
        frame_path = str(frame_path)
        self.fps = fps

        self._conf = {"codec": "libx264",
                      "fps": self.fps,
                      "audio_codec": "aac",
                      "temp_audiofile": "temp-audio.m4a",
                      "remove_temp": True}

        self._conf = {"bitrate": "5000k",
                      "fps": self.fps}

        # Load videos
        # video = mp.VideoFileClip(video1_path, audio=False)
        frames = [os.path.join(frame_path, x) for x in sorted(os.listdir(frame_path))]
        video = mp.ImageSequenceClip(frames, fps=fps)
        # in case
        # video1 = video1.fx(vfx.mirror_x)
        self.video = video
        self.duration = video.duration

    def add_text(self, text):
        video_text = mp.TextClip(text,
                                 # font="Amiri-Regular", # 'Amiri - regular',
                                 font='Amiri',
                                 color='white',
                                 method='caption',
                                 align="center",
                                 size=(self.video.w, None),
                                 fontsize=30)
        video_text = video_text.on_color(size=(self.video.w, video_text.h + 5),
                                         color=(0, 0, 0),
                                         col_opacity=0.6)
        # video_text = video_text.set_pos('bottom')
        video_text = video_text.set_pos('top')

        self.video = mp.CompositeVideoClip([self.video, video_text])

    def save(self, out_path):
        out_path = str(out_path)
        self.video.subclip(0, self.duration).write_videofile(out_path, **self._conf)

def is_permutation(a, b):
    """Checks if string is a permuation of one another

    Args:
        a (string): string a
        b (string): string b

    Returns:
        bool: If the strings have the same characters in different order
    """
    return len(a) == len(b) and Counter(a) == Counter(b)

def get_typename(fdname_approx):
    """Correct the folder names for the std and mean 
    for different datatypes (maybe mixed order)

    Args:
        fdname_approx (string): Approximate folder name 

    Returns:
        string: Corrected path given the approximate path 
    """
    from hydra.utils import get_original_cwd
    tmp_pth = fdname_approx[:-1]
    tmp_pth = tmp_pth[tmp_pth.index('deps'):]
    dtypes_avail = [pth.split('/')[-1] for pth in glob.glob(get_original_cwd() + '/'
                                                            + '/'.join(tmp_pth)
                                                            + '/*')]

    check_pths = [dtype for dtype in dtypes_avail if is_permutation(fdname_approx[-1], dtype)]
    return tmp_pth + [check_pths[0]]
