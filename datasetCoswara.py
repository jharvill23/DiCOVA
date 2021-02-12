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

        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        try:
            spect = self.get_feature(file)

            
            spect = spect.to(torch.float32)

        except:
            spect = None  # there are some weird ones with 1 frame causing problems

        return [file, spect]

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
        try:
            # files = [item[0] for item in data]
            # spects = [item[1] for item in data]
            # lengths = [item[2] for item in data]

            files = []
            spects = []
            for item in data:
                if item[1] != None:
                    files.append(item[0])
                    spects.append(item[1])

            features = pad_sequence(spects, batch_first=True, padding_value=0)

            # test_feature = features[0].detach().cpu().numpy()
            # plt.imshow(test_feature.T)
            # plt.show()

            return {'features': features, 'files': files, }
        except:
            return {'features': None, 'files': None}  # there are some weird ones with 1 frame causing problems




def main():
    """"""

if __name__ == "__main__":
    main()



