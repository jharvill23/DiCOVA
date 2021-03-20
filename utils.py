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
import soundfile as sf
import pickle
from sklearn.metrics import auc
import sys
import sox
import copy
import string
import opensmile

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

class Spec_OpenSMILE_augmentor(object):
    def __init__(self):
        """"""
        self.config = config
        self.dataloader_temp_wavs = self.config.directories.dataloader_temp_wavs
        self.opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def time_domain_spec_aug(self, audio, num_time_masks=2):
        num_audio_samples = len(audio)
        compressor = sox.Transformer()
        expander = sox.Transformer()
        """Step 1: time warping"""
        speed_up_factor = np.random.uniform(low=1.01, high=1.3)
        compressor.tempo(factor=speed_up_factor)
        slow_down_factor = np.random.uniform(low=0.7, high=0.99)
        expander.tempo(factor=slow_down_factor)
        # split audio into two pieces then choose randomly which piece to speed up or slow down
        split_point = int(num_audio_samples/2)
        first_half = audio[0:split_point]
        second_half = audio[split_point:]
        decision = np.random.uniform(low=0, high=1)
        if decision < 0.5:
            new_first_half = compressor.build_array(input_array=first_half, sample_rate_in=self.config.data.sr)
            new_second_half = expander.build_array(input_array=second_half, sample_rate_in=self.config.data.sr)
        else:
            new_first_half = expander.build_array(input_array=first_half, sample_rate_in=self.config.data.sr)
            new_second_half = compressor.build_array(input_array=second_half, sample_rate_in=self.config.data.sr)
        warped_audio = np.concatenate((new_first_half, new_second_half))
        num_audio_samples = len(warped_audio)
        # plt.subplot(211)
        # plt.plot(audio)
        # plt.subplot(212)
        # plt.plot(warped_audio)
        # plt.show()
        #
        # sf.write('warped_dummy.wav', warped_audio, self.config.data.sr, subtype='PCM_16')

        """Step 2: time masks. This equates to simple amplitude modulation."""
        min_time_mask_duration = 0.05*self.config.data.sr
        max_time_mask_duration = 0.55*self.config.data.sr
        fade_out_samples = 20
        if num_audio_samples > max_time_mask_duration + 10:
            for _ in range(num_time_masks):
                mask = np.ones_like(warped_audio)
                mask_start = np.random.randint(low=0, high=num_audio_samples-max_time_mask_duration)
                mask_dur = np.random.randint(low=min_time_mask_duration, high=max_time_mask_duration)
                mask_chunk = self.get_fade_out(mask_length=mask_dur, fade_out_samples=fade_out_samples)
                mask[mask_start:mask_start+mask_dur] = mask_chunk
                # old_warped = copy.deepcopy(warped_audio)
                warped_audio = warped_audio*mask
                # plt.subplot(311)
                # plt.plot(old_warped)
                # plt.subplot(312)
                # plt.plot(mask)
                # plt.subplot(313)
                # plt.plot(warped_audio)
                # plt.show()
        time_masked_audio = warped_audio
        """Step 3: Bandreject"""
        reject_frequency = np.random.uniform(low=80, high=int(self.config.data.sr/2)-300)
        width_q = np.random.uniform(low=0.05, high=0.1)
        bandreject = sox.Transformer()
        bandreject.bandreject(frequency=reject_frequency, width_q=width_q)
        bandrejected_audio = bandreject.build_array(input_array=time_masked_audio, sample_rate_in=self.config.data.sr)
        # sf.write('fully_processed_dummy.wav', bandrejected_audio, self.config.data.sr, subtype='PCM_16')
        return bandrejected_audio

    def get_fade_out(self, mask_length, fade_out_samples):
        start = np.linspace(start=1, stop=0, num=fade_out_samples)
        middle = np.zeros(shape=(mask_length-2*fade_out_samples))
        stop = np.linspace(start=0, stop=1, num=fade_out_samples)
        mask = np.concatenate((start, middle, stop))
        return mask

    def augment(self, filename, audio, augment_number, dump_dir):
        new_audio = self.time_domain_spec_aug(audio=audio)

        """Compute spectrogram"""
        feature_processor = Mel_log_spect()
        spectrogram = feature_processor.get_Mel_log_spect(new_audio)

        """Get OpenSMILE features"""
        # Generate random filename of length 24 to write to disk
        temp_write_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=24))
        dump_path = os.path.join(self.dataloader_temp_wavs, temp_write_name + '.wav')
        sf.write(dump_path, new_audio, self.config.data.sr, subtype='PCM_16')
        output = self.opensmile.process_file(dump_path)
        os.remove(dump_path)  # don't want to keep all those temporary files!
        opensmile_feats = np.squeeze(output.to_numpy())

        """Organize and save to disk"""
        features = {'spectrogram': spectrogram, 'opensmile': opensmile_feats}
        permanent_dump_path = os.path.join(dump_dir, filename + '_' + str(augment_number) + '.pkl')
        joblib.dump(value=features, filename=permanent_dump_path)

def process_augmentor(data):
    file = data['file']
    filename = file.split('/')[-1][:-4]
    dump_dir = data['dump_dir']
    augmentor = Spec_OpenSMILE_augmentor()
    audio, _ = librosa.core.load(file, sr=config.data.sr)
    for i in range(config.data.num_augment_examples):
        try:
            augmentor.augment(filename=filename, audio=audio, augment_number=i, dump_dir=dump_dir)
        except:
            print("Had trouble with audio file...")

def augment_dicova():
    files = collect_files(config.directories.dicova_wavs)
    dump_dir = config.directories.dicova_augmented_feats
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    new_list = []
    for file in files:
        new_list.append({'file': file, 'dump_dir': dump_dir})
    # process_augmentor(new_list[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process_augmentor, new_list)):
            """"""

def get_normalization_factors_opensmile():
    print('Getting normalization factors for dicova opensmile...')
    files = collect_files(config.directories.dicova_opensmile_feats)
    feats = np.zeros(shape=(len(files), 6373))
    for i, file in enumerate(files):
        row = joblib.load(file)
        feats[i] = row
    max_vals = np.max(np.abs(feats), axis=0)
    joblib.dump(max_vals, 'dicova_opensmile_maxvals.pkl')

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

def flac2wav(filelist, dump_dir):
    for file in tqdm(filelist):
        audio, sr = librosa.core.load(file, sr=44100)
        """Save the data as a wav file instead of flac"""
        name = file.split('/')[-1][:-5] + '.wav'
        dump_path = os.path.join(dump_dir, name)
        x = np.round(audio * 32767)
        x = x.astype('int16')
        sf.write(dump_path, x, sr, subtype='PCM_16')

def process(data):
    file = data['file']
    dump_dir = data['dump_dir']
    try:
        audio, _ = librosa.core.load(file, sr=config.data.sr)
        feature_processor = Mel_log_spect()
        features = feature_processor.get_Mel_log_spect(audio)
        # plt.imshow(features.T)
        # plt.show()
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

def scoring(refs, sys_outs, out_file):
    """
        inputs::
        refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
        sys_outs: a txt file with a list of scores (probability of being covid positive) for each wav-fileid in the format: <id> <score>
        threshold (optional): a np.array(), like np.arrange(0,1,.01), sweeping for AUC

        outputs::

        """

    thresholds = np.arange(0, 1, 0.01)
    # Read the ground truth labels into a dictionary
    data = open(refs).readlines()
    reference_labels = {}
    categories = ['n', 'p']
    for line in data:
        key, val = line.strip().split()
        reference_labels[key] = categories.index(val)

    # Read the system scores into a dictionary
    data = open(sys_outs).readlines()
    sys_scores = {}
    for line in data:
        key, val = line.strip().split()
        sys_scores[key] = float(val)
    del data

    # Ensure all files in the reference have system scores and vice-versa
    if len(sys_scores) != len(reference_labels):
        print("Expected the score file to have scores for all files in reference and no duplicates/extra entries")
        return None
    # %%

    # Arrays to store true positives, false positives, true negatives, false negatives
    TP = np.zeros((len(reference_labels), len(thresholds)))
    TN = np.zeros((len(reference_labels), len(thresholds)))
    keyCnt = -1
    for key in sys_scores:  # Repeat for each recording
        keyCnt += 1
        sys_labels = (sys_scores[key] >= thresholds) * 1  # System label for a range of thresholds as binary 0/1
        gt = reference_labels[key]

        ind = np.where(sys_labels == gt)  # system label matches the ground truth
        if gt == 1:  # ground-truth label=1: True positives
            TP[keyCnt, ind] = 1
        else:  # ground-truth label=0: True negatives
            TN[keyCnt, ind] = 1

    total_positives = sum(reference_labels.values())  # Total number of positive samples
    total_negatives = len(reference_labels) - total_positives  # Total number of negative samples

    TP = np.sum(TP, axis=0)  # Sum across the recordings
    TN = np.sum(TN, axis=0)

    TPR = TP / total_positives  # True positive rate: #true_positives/#total_positives
    TNR = TN / total_negatives  # True negative rate: #true_negatives/#total_negatives

    AUC = auc(1 - TNR, TPR)  # AUC

    ind = np.where(TPR >= 0.8)[0]
    sensitivity = TPR[ind[-1]]
    specificity = TNR[ind[-1]]

    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except AUC) is a array for different threshold values
    # Specificity at 90% sensitivity
    scores = {'TPR': TPR,
              'FPR': 1 - TNR,
              'AUC': AUC,
              'sensitivity': sensitivity,
              'specificity': specificity,
              'thresholds': thresholds}

    with open(out_file, "wb") as f:
        pickle.dump(scores, f)

def summary(folname, scores, iterations):
    # folname = sys.argv[1]
    num_files = 1
    R = []
    for i in range(num_files):
        # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
        # res = pickle.load(open(scores))
        res = joblib.load(scores)
        R.append(res)

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    for i in range(num_files):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
        plt.plot(data_x[i], data_y[i], label='V-' + str(i + 1) + ', auc=' + str(np.round(data_auc[i], 2)), c=clr_1,
                 alpha=0.2)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot_' + str(iterations) + '.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")
    return np.round(np.mean(np.array(data_auc)), 2)

def eval_summary(folname, outfiles):
    # folname = sys.argv[1]
    num_files = len(outfiles)
    R = []
    for file in outfiles:
        # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
        # res = pickle.load(open(scores))
        res = joblib.load(file)
        R.append(res)

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    for i in range(num_files):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
        plt.plot(data_x[i], data_y[i], label='V-' + str(i + 1) + ', auc=' + str(np.round(data_auc[i], 2)), c=clr_1,
                 alpha=0.2)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")

def best_config_results_plot(folname, outfiles):
    # folname = sys.argv[1]
    num_files = len(outfiles)
    R = []
    for file in outfiles:
        # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
        # res = pickle.load(open(scores))
        res = joblib.load(file)
        R.append(res)

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    for i in range(num_files):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
        plt.plot(data_x[i], data_y[i], label='V-' + str(i + 1) + ', auc=' + str(np.round(data_auc[i], 2)), c=clr_1,
                 alpha=0.2)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")


def main():
    """"""
    get_normalization_factors_opensmile()
    # augment_dicova()
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

