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

class DiCOVA(object):
    """Solver"""

    def __init__(self, config):
        self.config = config
        self.root = self.config.directories.dicova_root
        self.metadata = self.get_metadata()
        self.partition = self.get_partition()
        self.featdir = self.config.directories.dicova_logspect_feats

    def get_partition(self):
        if not os.path.exists('dicova_partitions.pkl'):
            files = collect_files(os.path.join(self.config.directories.dicova_root, 'LISTS'))
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
                train_files = [x.strip() + '.wav' for x in train_files]
                """Get train positives and train negatives"""
                train_pos = []
                train_neg = []
                for file in train_files:
                    meta = self.get_file_metadata(file)
                    if meta['Covid_status'] == 'p':
                        train_pos.append(file)
                    elif meta['Covid_status'] == 'n':
                        train_neg.append(file)
                val = partition['val']
                with open(val) as f:
                    val_files = f.readlines()
                val_files = [x.strip() + '.wav' for x in val_files]
                val_pos = []
                val_neg = []
                for file in val_files:
                    meta = self.get_file_metadata(file)
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

    def feature_path_partition(self):
        new_dict = {}
        for fold, fold_dictionary in self.partition.items():
            new_dict[fold] = {}
            for type_key, type_list in fold_dictionary.items():
                new_list = [os.path.join(self.featdir, x[:-4] + '.pkl') for x in type_list]
                new_dict[fold][type_key] = new_list
        self.feat_partition = new_dict

    def get_metadata(self):
        if not os.path.exists('dicova_metadata.pkl'):
            import csv
            lines = []
            with open(os.path.join(self.config.directories.dicova_root, 'metadata.csv'), newline='') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 0:
                        lines.append(row)
            metadata = {}
            for line in lines:
                metadata[line[0]] = {'Covid_status': line[1], 'Gender': line[2], 'Nationality': line[3]}
            joblib.dump(metadata, 'dicova_metadata.pkl')
        else:
            metadata = joblib.load('dicova_metadata.pkl')
        return metadata

    def get_file_metadata(self, file):
        filename = file.split('/')[-1][:-4]
        return self.metadata[filename]

    def get_features(self):
        """Add full filepaths"""
        partition = self.partition['1']
        filelist = partition['train_pos'] + partition['train_neg'] + partition['val_pos'] + partition['val_neg']
        new_filelist = []
        for file in filelist:
            new_filelist.append(os.path.join(self.root, 'AUDIO', file[:-4] + '.flac'))
        dump_dir = self.config.directories.dicova_wavs
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        utils.flac2wav(filelist=new_filelist, dump_dir=dump_dir)
        new_filelist = utils.collect_files(self.config.directories.dicova_wavs)
        utils.get_features(filelist=new_filelist, dump_dir=self.featdir)

class Dataset(object):
    """Solver"""

    def __init__(self, config, params):
        """Get the data and supporting files"""
        self.config = config
        self.feature_dir = self.config.directories.freesound_logspect_feats
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        self.data_object = params['data_object']
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.incorrect_scaler = self.config.post_pretraining_classifier.incorrect_scaler

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        metadata = self.data_object.get_file_metadata(file)
        label = self.class2index[metadata['Covid_status']]
        label = self.to_GPU(torch.from_numpy(np.asarray(label)))
        feats = self.to_GPU(torch.from_numpy(joblib.load(file)))
        """Get incorrect_scaler value"""
        if metadata['Covid_status'] == 'p':
            scaler = self.incorrect_scaler
        else:
            scaler = 1
        scaler = self.to_GPU(torch.from_numpy(np.asarray(scaler)))
        scaler = scaler.to(torch.float32)
        scaler.requires_grad = True
        return file, feats, label, scaler

    def to_GPU(self, tensor):
        if self.config.use_gpu == True:
            tensor = tensor.cuda()
            return tensor
        else:
            return tensor

    def collate(self, data):
        files = [item[0] for item in data]
        spects = [item[1] for item in data]
        labels = [item[2] for item in data]
        scalers = [item[3] for item in data]
        spects = pad_sequence(spects, batch_first=True, padding_value=0)
        labels = torch.stack([x for x in labels])
        scalers = torch.stack([x for x in scalers])
        return {'files': files, 'features': spects, 'labels': labels, 'scalers': scalers}


def main():
    config = get_config.get()
    dicova = DiCOVA(config=config)
    dicova.get_features()

if __name__ == "__main__":
    main()