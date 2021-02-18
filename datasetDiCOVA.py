import torchaudio
import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
import shutil
from torch.utils import data
from itertools import groupby
import json
from Levenshtein import distance as levenshtein_distance
import multiprocessing
import concurrent.futures
import random
from torch.utils import data
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import utils
import librosa

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))


class Dataset(object):
    """Solver"""

    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.feature_dir = config.directories.opensmile_feats
        self.config = config
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        if os.path.exists('dicova_metadata.pkl'):
            self.metadata = joblib.load('dicova_metadata.pkl')
        else:
            self.metadata = utils.dicova_metadata()
        self.triplet = params["triplet"]
        self.files_dict = params["files_dict"]
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.incorrect_scaler = config.model.incorrect_scaler

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        metadata = utils.dicova_metadata(file, metadata=self.metadata)
        if self.triplet:
            """Our file is the anchor. Now we need a positive and negative """
            random.shuffle(self.files_dict['positive'])
            random.shuffle(self.files_dict['negative'])
            if metadata['Covid_status'] == 'p':
                pos = self.files_dict['positive'][0]
                neg = self.files_dict['negative'][0]
            else:
                neg = self.files_dict['positive'][0]
                pos = self.files_dict['negative'][0]
            triplet_files = {'anchor': file, 'pos': pos, 'neg': neg}
            triplet_features = {}
            triplet_features['anchor'] = self.to_GPU(torch.from_numpy(joblib.load(triplet_files['anchor'])))
            triplet_features['pos'] = self.to_GPU(torch.from_numpy(joblib.load(triplet_files['pos'])))
            triplet_features['neg'] = self.to_GPU(torch.from_numpy(joblib.load(triplet_files['neg'])))

            for key, value in triplet_features.items():
                triplet_features[key] = value.to(torch.float32)

            return [triplet_files, triplet_features]
        else:
            feature = self.to_GPU(torch.from_numpy(joblib.load(file)))
            label = np.asarray(self.class2index[metadata['Covid_status']])
            """Get incorrect_scaler value"""
            if metadata['Covid_status'] == 'p':
                scaler = self.incorrect_scaler
            else:
                scaler = 1
            scaler = np.asarray(scaler)
            # scaler = np.ones_like(label)
            # positive_indices = np.where(label == self.class2index['p'])
            # scaler[positive_indices] = self.incorrect_scaler
            scaler = torch.from_numpy(scaler)
            scaler = self.to_GPU(scaler)
            scaler = scaler.to(torch.float32)
            scaler.requires_grad = True
            label = torch.from_numpy(label)
            label = self.to_GPU(label)
            return [file, feature, label, scaler]

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def require_grad(self, tensor_list):
        new_list = []
        for tensor in tensor_list:
            tensor.requires_grad = True
            new_list.append(tensor)
        return new_list

    def get_feature(self, file):
        metadata = utils.get_metadata(file)
        filename = metadata['filename']
        feature_path = os.path.join(self.feature_dir, filename + '.pkl')
        try:
            tensor = joblib.load(feature_path)
            tensor = self.to_GPU(tensor)
        except:
            audio, sr = librosa.core.load(path=file, sr=self.sr)
            audio = torch.from_numpy(audio)
            audio = self.to_GPU(audio)
            audio = audio.to(torch.float32)
            tensor = self.dB(self.mel_spect(audio))
            tensor = (tensor - torch.min(tensor))
            tensor = tensor / torch.max(tensor)
            tensor = torch.squeeze(tensor)
            tensor = tensor.T
            joblib.dump(tensor, feature_path)
        return tensor

    def collate(self, data):
        # try:
            # files = [item[0] for item in data]
            # spects = [item[1] for item in data]
            # lengths = [item[2] for item in data]
            if self.triplet:
                triplet_files = [item[0] for item in data]
                triplet_features = [item[1] for item in data]

                anchors = torch.stack([x['anchor'] for x in triplet_features])
                pos = torch.stack([x['pos'] for x in triplet_features])
                neg = torch.stack([x['neg'] for x in triplet_features])

                return {'anchors': anchors, 'pos': pos, 'neg': neg, 'files': triplet_files}
            else:
                files = [item[0] for item in data]
                features = [item[1] for item in data]
                labels = [item[2] for item in data]
                scalers = [item[3] for item in data]

                features = torch.stack([x for x in features])
                labels = torch.stack([x for x in labels])
                scalers = torch.stack([x for x in scalers])
                return {'files': files, 'features': features, 'labels': labels, 'scalers': scalers}
        # except:
        #     return {'features': None, 'files': None}  # there are some weird ones with 1 frame causing problems


def main():
    """"""


if __name__ == "__main__":
    main()



