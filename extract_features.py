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


config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def make_dirs(files):
    """"""
    for file in tqdm(files):
        metadata = utils.get_metadata(file)
        if metadata['isWav']:
            collection_set = os.path.join(root, metadata['collection_set'])
            id = os.path.join(root, metadata['collection_set'], metadata['id'])
            for dir_ in [collection_set, id]:
                if not os.path.isdir(dir_):
                    os.mkdir(dir_)

def process(file):
    """"""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        metadata = utils.get_metadata(file)
        if metadata['isWav']:
            dump_path = os.path.join(root, metadata['collection_set'], metadata['id'], metadata['filename'] + '.pkl')
            y = smile.process_file(metadata['wav_loc'])
            features = np.squeeze(y.to_numpy())
            joblib.dump(features, dump_path)
    except:
        """"""


files = utils.collect_files(config.directories.coswara_root)
root = config.directories.opensmile_feats
if not os.path.isdir(root):
    os.mkdir(root)
make_dirs(files)

# process(files[0])
with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    for _ in tqdm(executor.map(process, files)):
        """"""