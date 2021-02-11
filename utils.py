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


config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def collect_files(directory):
    all_files = []
    for path, subdirs, files in tqdm(os.walk(directory)):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)
    return all_files

def get_metadata(file):
    pieces = file.split('/')
    wav_loc = os.path.join(config.directories.coswara_root, pieces[-3], pieces[-2], pieces[-1])
    collection_set = pieces[-3]
    id = pieces[-2]
    filename = pieces[-1][:-4]
    if pieces[-1][-4:] == '.wav':
        isWav = True
    else:
        isWav = False
    return {'collection_set': collection_set, 'id': id, 'filename': filename,
            'wav_loc': wav_loc, 'isWav': isWav}

