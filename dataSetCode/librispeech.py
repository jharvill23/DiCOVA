from pydub import AudioSegment
import os
from easydict import EasyDict as edict
import yaml
from config import get_config
import utils
import joblib
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import collect_files
import numpy as np
import espnet
import espnet.transform as trans
import espnet.transform.spec_augment as SPEC
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# print(os.getcwd())
#
# file = '/home/john/Documents/School/Spring_2021/COUGHVID/public_dataset/0a1b4119-cc22-4884-8f0f-34e8207c31d1.webm'
#
# song = AudioSegment.from_file(file, "webm")
# print("Loaded")
# print(os.getcwd())
#
# song.export("DUMDUM.wav", format="wav")


class LibriSpeech(object):
    """Solver"""

    def __init__(self, config):
        self.config = config
        self.root = self.config.directories.librispeech_root
        self.librispeech_wavs = self.config.directories.librispeech_wavs
        self.librispeech_feats = self.config.directories.librispeech_logspect_feats
        # self.metadata = self.get_metadata()
        # self.partition = self.get_partition()
        # self.featdir = self.config.directories.dicova_logspect_feats

    def get_partition(self):
        if not os.path.exists('librispeech_partition.pkl'):
            files = collect_files(self.config.directories.librispeech_logspect_feats)
            random.shuffle(files)
            split_index = round(0.98*len(files))
            train = files[0:split_index]
            val = files[split_index:]
            partition = {'train': train, 'test': val}
            joblib.dump(partition, 'librispeech_partition.pkl')
        else:
            partition = joblib.load('librispeech_partition.pkl')
        self.partition = partition

    def feature_path_partition(self):
        new_dict = {}
        for fold, fold_dictionary in self.partition.items():
            new_dict[fold] = {}
            for type_key, type_list in fold_dictionary.items():
                new_list = [os.path.join(self.featdir, x[:-4] + '.pkl') for x in type_list]
                new_dict[fold][type_key] = new_list
        self.feat_partition = new_dict

    def get_metadata(self):
        """"""
        # if not os.path.exists('coughvid_metadata.pkl'):
        #     import csv
        #     lines = []
        #     with open(os.path.join(self.config.directories.coughvid_root, 'metadata_compiled.csv'), newline='') as f:
        #         reader = csv.reader(f)
        #         for i, row in enumerate(reader):
        #             if i > 0:
        #                 lines.append(row)
        #     metadata = {}
        #     for line in lines:
        #         metadata[line[0]] = {'Covid_status': line[1], 'Gender': line[2], 'Nationality': line[3]}
        #     joblib.dump(metadata, 'dicova_metadata.pkl')
        # else:
        #     metadata = joblib.load('dicova_metadata.pkl')
        # return metadata

    def get_file_metadata(self, file):
        filename = file.split('/')[-1][:-4]
        return self.metadata[filename]

    def flac_to_wav(self):
        if not os.path.isdir(self.librispeech_wavs):
            os.mkdir(self.librispeech_wavs)
        files = utils.collect_files(self.root)
        new_files = []
        for file in files:
            if file[-5:] == '.flac':
                new_files.append(file)
        utils.flac2wav(new_files, dump_dir=self.librispeech_wavs, sr=16000)

    def get_features(self):
        """Add full filepaths"""
        filelist = utils.collect_files(self.librispeech_wavs)
        dump_dir = self.librispeech_feats
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        utils.get_features(filelist=filelist, dump_dir=dump_dir)

class Dataset(object):
    """Solver"""

    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        self.feature_dir = self.config.directories.librispeech_logspect_feats
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.data_object = params['data_object']
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler
        self.specaugment = params['specaugment']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        if self.specaugment:
            feats = joblib.load(file)
            time_width = round(feats.shape[0]*0.1)
            aug_feats = SPEC.spec_augment(feats, resize_mode='PIL', max_time_warp=80,
                                                                   max_freq_width=20, n_freq_mask=1,
                                                                   max_time_width=time_width, n_time_mask=2,
                                                                   inplace=False, replace_with_zero=True)
            feats = self.to_GPU(torch.from_numpy(aug_feats))
        else:
            feats = self.to_GPU(torch.from_numpy(joblib.load(file)))

        return file, feats

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        return {'files': files, 'features': spects}



def main():
    config = get_config.get()
    librispeech = LibriSpeech(config=config)
    # librispeech.flac_to_wav()
    librispeech.get_features()

if __name__ == "__main__":
    main()