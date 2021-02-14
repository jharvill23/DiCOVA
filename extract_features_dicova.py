import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
import pandas as pd
import shutil
from torch.utils import data
from itertools import groupby
import json
import random
import librosa
import matplotlib.pyplot as plt
import time
import multiprocessing
import concurrent.futures
import utils
import opensmile
import pandas as pd
import soundfile as sf


config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

STEPS = '1'

files = utils.collect_files(os.path.join(config.directories.dicova_root, 'AUDIO'))
root = config.directories.opensmile_feats
if not os.path.isdir(root):
    os.mkdir(root)

wavs = config.directories.dicova_wavs
if not os.path.isdir(wavs):
    os.mkdir(wavs)

if '0' in STEPS:
    """First convert all .flac files to .wav"""
    for file in tqdm(files):
        audio, sr = librosa.core.load(file, sr=44100)
        """Save the data as a wav file instead of flac"""
        name = file.split('/')[-1][:-5] + '.wav'
        dump_path = os.path.join(config.directories.dicova_wavs, name)
        x = np.round(audio * 32767)
        x = x.astype('int16')
        # plt.plot(x)
        # plt.show()
        sf.write(dump_path, x, sr, subtype='PCM_16')

files = utils.collect_files(config.directories.dicova_wavs)

def process(file):
    """"""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        name = file.split('/')[-1][:-4]
        dump_path = os.path.join(root, name + '.pkl')
        y = smile.process_file(file)
        features = np.squeeze(y.to_numpy())
        joblib.dump(features, dump_path)
    except:
        """"""

if '1' in STEPS:
    # process(files[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process, files)):
            """"""