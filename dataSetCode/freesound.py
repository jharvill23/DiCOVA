import os
from easydict import EasyDict as edict
import yaml
from config import get_config
import utils
import joblib
import torch
from torch.nn.utils.rnn import pad_sequence

class FreeSound(object):
    """Solver"""

    def __init__(self, config):
        self.config = config
        self.root = self.config.directories.freesound_root
        self.metadata = self.get_metadata()
        self.partition = self.get_partition()
        self.featdir = self.config.directories.freesound_logspect_feats
        self.train_wavs = os.path.join(self.root, 'FSDKaggle2018.audio_train')
        self.test_wavs = os.path.join(self.root, 'FSDKaggle2018.audio_test')

    def get_partition(self):
        file_root = os.path.join(self.root, 'FSDKaggle2018.meta')
        train_file = os.path.join(file_root, 'train_post_competition.csv')
        test_file = os.path.join(file_root, 'test_post_competition_scoring_clips.csv')

        train_files = []
        test_files = []
        for i, file in enumerate([train_file, test_file]):
            file1 = open(file, 'r')
            """Structure of train file is [fname, label, manually_verified, id, license]"""
            index = 0
            for line in file1:
                line = line[:-1]  # remove newline character
                pieces = line.split(',')
                if index > 0:
                    if i == 0:
                        train_files.append(pieces[0])
                    elif i == 1:
                        test_files.append(pieces[0])
                index += 1
            file1.close()
        partition = {'train': train_files, 'test': test_files}
        return partition

    def feature_path_partition(self):
        train_files = []
        test_files = []
        for file in self.partition['train']:
            train_files.append(os.path.join(self.featdir, file[:-4] + '.pkl'))
        for file in self.partition['test']:
            test_files.append(os.path.join(self.featdir, file[:-4] + '.pkl'))
        self.feat_partition = {'train': train_files, 'test': test_files}

    def get_metadata(self):
        file_root = os.path.join(self.root, 'FSDKaggle2018.meta')
        train_file = os.path.join(file_root, 'train_post_competition.csv')
        test_file = os.path.join(file_root, 'test_post_competition_scoring_clips.csv')

        metadata = {}
        for file in [train_file, test_file]:
            file1 = open(file, 'r')
            """Structure of train file is [fname, label, manually_verified, id, license]"""
            index = 0
            for line in file1:
                line = line[:-1]  # remove newline character
                file_data = {}
                pieces = line.split(',')
                if index > 0:
                    file_data['label'] = pieces[1]
                    file_data['manually_verified'] = pieces[2]
                    file_data['id'] = pieces[3]
                    file_data['license'] = pieces[4]
                    metadata[pieces[0][:-4]] = file_data
                index += 1
            file1.close()
        return metadata

    def get_features(self):
        # filelist = self.partition['train'] + self.partition['test']
        """Add full filepaths"""
        filelist = []
        for file in self.partition['train']:
            filelist.append(os.path.join(self.train_wavs, file))
        for file in self.partition['test']:
            filelist.append(os.path.join(self.test_wavs, file))
        dump_dir = self.featdir
        utils.get_features(filelist=filelist, dump_dir=dump_dir)

class Dataset(object):
    """Solver"""

    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        self.feature_dir = self.config.directories.freesound_logspect_feats
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
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
    freesound = FreeSound(config=config)
    freesound.get_features()

if __name__ == "__main__":
    main()