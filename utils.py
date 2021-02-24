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
import pandas as pd
from config import get_config

config = get_config.get()

class Mel_log_spect(object):
    def __init__(self):
        self.nfft = config.data.fftl
        self.num_mels = config.data.num_mels
        self.hop_length = config.data.hop_length
        self.top_db = config.data.top_db
        self.sr = config.data.sr

    def feature_normalize(self, x):
        log_min = np.min(x)
        x = x - log_min
        x = x / self.top_db
        x = x.T
        return x

    def get_Mel_log_spect(self, y):
        y = librosa.util.normalize(S=y)
        spect = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.nfft,
                                               hop_length=self.hop_length, n_mels=self.num_mels)
        log_spect = librosa.core.amplitude_to_db(spect, ref=1.0, top_db=self.top_db)
        log_spect = self.feature_normalize(log_spect)
        return log_spect

    def norm_Mel_log_spect_to_amplitude(self, feature):
        feature = feature * self.top_db
        spect = librosa.core.db_to_amplitude(feature, ref=1.0)
        return spect

    def audio_from_spect(self, feature):
        spect = self.norm_Mel_log_spect_to_amplitude(feature)
        audio = librosa.feature.inverse.mel_to_audio(spect.T, sr=self.sr, n_fft=self.nfft, hop_length=self.hop_length)
        return audio

    def convert_and_write(self, load_path, write_path):
        y, sr = librosa.core.load(path=load_path, sr=self.sr)
        feature = self.get_Mel_log_spect(y, n_mels=self.num_mels)
        audio = self.audio_from_spect(feature)
        librosa.output.write_wav(write_path, y=audio, sr=self.sr, norm=True)

def collect_files(directory):
    all_files = []
    for path, subdirs, files in tqdm(os.walk(directory)):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)
    return all_files

def dicova_metadata(file, metadata=None):
    name = file.split('/')[-1][:-4]
    if not os.path.exists('dicova_metadata.pkl'):
        import csv
        lines = []
        with open(os.path.join(config.directories.dicova_root, 'metadata.csv'), newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i > 0:
                    lines.append(row)
        metadata = {}
        for line in lines:
            metadata[line[0]] = {'Covid_status': line[1], 'Gender': line[2], 'Nationality': line[3]}
        joblib.dump(metadata, 'dicova_metadata.pkl')
    else:
        if metadata == None:
            metadata = joblib.load('dicova_metadata.pkl')
    sub_data = metadata[name]
    sub_data['name'] = name
    return sub_data

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

def get_dicova_partitions():
    if not os.path.exists('dicova_partitions.pkl'):
        files = collect_files(os.path.join(config.directories.dicova_root, 'LISTS'))
        folds = {}
        for file in files:
            name = file.split('/')[-1][:-4]
            pieces = name.split('_')
            train_val = pieces[0]
            fold = pieces[2]
            if fold in folds:
                folds[fold][train_val] = file
            else:
                folds[fold] = {train_val: file}
        fold_files = {}
        for fold, partition in folds.items():
            train = partition['train']
            with open(train) as f:
                train_files = f.readlines()
            train_files = [os.path.join(config.directories.opensmile_feats, x.strip() + '.pkl') for x in train_files]
            """Get train positives and train negatives"""
            train_pos = []
            train_neg = []
            for file in train_files:
                meta = dicova_metadata(file)
                if meta['Covid_status'] == 'p':
                    train_pos.append(file)
                elif meta['Covid_status'] == 'n':
                    train_neg.append(file)
            val = partition['val']
            with open(val) as f:
                val_files = f.readlines()
            val_files = [os.path.join(config.directories.opensmile_feats, x.strip() + '.pkl') for x in val_files]
            val_pos = []
            val_neg = []
            for file in val_files:
                meta = dicova_metadata(file)
                if meta['Covid_status'] == 'p':
                    val_pos.append(file)
                elif meta['Covid_status'] == 'n':
                    val_neg.append(file)
            fold_files[fold] = {'train_pos': train_pos, 'train_neg': train_neg,
                                'val_pos': val_pos, 'val_neg': val_neg}
        joblib.dump(fold_files, 'dicova_partitions.pkl')
    else:
        fold_files = joblib.load('dicova_partitions.pkl')
    return fold_files

def get_class2index_and_index2class():
    class2index = {'p': 0, 'n': 1}
    index2class = {0: 'p', 1: 'n'}
    return class2index, index2class

def process(data):
    file = data['file']
    dump_dir = data['dump_dir']
    try:
        audio, _ = librosa.core.load(file, sr=config.data.sr)
        feature_processor = Mel_log_spect()
        features = feature_processor.get_Mel_log_spect(audio)
        dump_path = os.path.join(dump_dir, file.split('/')[-1][:-4] + '.pkl')
        joblib.dump(features, dump_path)
    except:
        print("Had trouble processing file " + file + " ...")

def get_features(filelist, dump_dir):
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    new_list = []
    for file in filelist:
        new_list.append({'file': file, 'dump_dir': dump_dir})
    # process(new_list[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process, new_list)):
            """"""



def main():
    """"""
    # files = get_dicova_partitions()
    # meta = dicova_metadata('/home/john/Documents/School/Spring_2021/DiCOVA/wavs/aBXnKRBt_cough.wav')
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

