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
import json


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

    """We now want to load the metadata file"""
    try:
        metadata_path = os.path.join(config.directories.coswara_root, pieces[-3], pieces[-2], 'metadata.json')
        with open(metadata_path) as f:
            metadata = json.load(f)
        covid_status = metadata['covid_status']
    except:
        covid_status = None

    return {'collection_set': collection_set, 'id': id, 'filename': filename,
            'wav_loc': wav_loc, 'isWav': isWav, 'covid_status': covid_status}

def get_coswara_partition():
    files = collect_files(config.directories.opensmile_feats)
    covid_positive = []
    covid_negative = []
    for file in files:
        meta = get_metadata(file)
        covid_status = meta['covid_status']
        if 'positive' in covid_status:
            covid_positive.append(file)
        else:
            covid_negative.append(file)
    random.shuffle(covid_positive)
    random.shuffle(covid_negative)
    num_test_utts = config.data.num_test_utts_per_class
    num_val_utts = config.data.num_val_utts_per_class
    test_positive = covid_positive[0:num_test_utts]
    val_positive = covid_positive[num_test_utts:num_test_utts + num_val_utts]
    train_positive = covid_positive[num_test_utts + num_val_utts:]
    test_negative = covid_negative[0:num_test_utts]
    val_negative = covid_negative[num_test_utts:num_test_utts + num_val_utts]
    train_negative = covid_negative[num_test_utts + num_val_utts:]
    partition = {'test_positive': test_positive,
                 'val_positive': val_positive,
                 'train_positive': train_positive,
                 'test_negative': test_negative,
                 'val_negative': val_negative,
                 'train_negative': train_negative}
    joblib.dump(partition, 'coswara_partition.pkl')


def main():
    """"""
    # get_coswara_partition()
    # files = collect_files(config.directories.opensmile_feats)
    # possible_covid_labels = []
    # for file in tqdm(files):
    #     meta = get_metadata(file)
    #     covid_status = meta['covid_status']
    #     if covid_status not in possible_covid_labels:
    #         possible_covid_labels.append(covid_status)
    """If 'positive' is in the covid_status, they have covid. Otherwise no."""

if __name__ == "__main__":
    main()

