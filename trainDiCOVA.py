import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import model
import yaml
from easydict import EasyDict as edict
import shutil
from utils import collect_files
import utils
from datasetDiCOVA import Dataset
from torch.utils import data
from itertools import groupby
import json
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import sklearn.metrics.pairwise as pairwise


config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'trial_5_classification_higher_weight_for_positive_30'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

FOLD = '1'
TRAIN = True
EVAL = False
PERFORMANCE = False
LOAD_MODEL = False

class Solver(object):
    """Solver"""

    def __init__(self):
        """Initialize configurations."""

        # Training configurations.
        self.g_lr = config.model.lr
        self.torch_type = torch.float32

        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')
        self.class2index, self.index2class = utils.get_class2index_and_index2class()
        self.incorrect_scaler = config.model.incorrect_scaler

        # Directories.
        self.log_dir = os.path.join(exp_dir, 'logs')
        self.model_save_dir = os.path.join(exp_dir, 'models')
        self.train_data_dir = config.directories.features
        self.predict_dir = os.path.join(exp_dir, 'predictions')
        self.images_dir = os.path.join(exp_dir, 'images')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.predict_dir):
            os.mkdir(self.predict_dir)
        if not os.path.isdir(self.images_dir):
            os.mkdir(self.images_dir)

        if not os.path.exists('dicova_partitions.pkl'):
            utils.get_dicova_partitions()

        """Partition file"""
        if TRAIN:  # only copy these when running a training session, not eval session
            # copy partition to exp_dir then use that for trial (just in case you change partition for other trials)
            shutil.copy(src='dicova_partitions.pkl', dst=os.path.join(exp_dir, 'dicova_partitions.pkl'))
            self.partition = os.path.join(exp_dir, 'dicova_partitions.pkl')
            # copy config as well
            shutil.copy(src='config.yml', dst=os.path.join(exp_dir, 'config.yml'))

        self.partition = os.path.join(exp_dir, 'dicova_partitions.pkl')

        # Step size.
        self.log_step = config.train.log_step
        self.model_save_step = config.train.model_save_step

        # Build the model
        self.build_model()
        if EVAL or LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        self.G = model.CTCmodel(config)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.print_network(self.G, 'G')
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self):
        """Restore the model"""
        print('Loading the trained models... ')
        G_path = './exps/trial_1_triplet_loss/models/15000-G.ckpt'
        g_checkpoint = self._load(G_path)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def update_lr(self, g_lr):
        """Decay learning rates of g"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def get_train_test(self):
        partition = joblib.load(self.partition)
        partition = partition[FOLD]

        train_files = {'positive': partition['train_pos'], 'negative': partition['train_neg']}
        # test_files = {'positive': partition['test_positive'], 'negative': partition['test_negative']}
        val_files = {'positive': partition['val_pos'], 'negative': partition['val_neg']}

        return train_files, val_files

    def val_loss(self, val, iterations):
        val_loss = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for batch_number, features in tqdm(enumerate(val)):
            # try:
                feature = features['features']
                labels = features['labels']
                files = features['files']
                scalers = features['scalers']
                info = utils.dicova_metadata(files[0])  # batch size 1
                self.G = self.G.eval()

                predictions = self.G(feature)

                loss = self.crossent_loss(predictions, labels)
                loss = loss * scalers
                val_loss += loss.item()

                predictions = np.squeeze(predictions.detach().cpu().numpy())
                pred_value = self.index2class[np.argmax(predictions)]

                if info['Covid_status'] == 'p':
                    if pred_value == 'p':
                        TP += 1
                    elif pred_value == 'n':
                        FN += 1
                elif info['Covid_status'] == 'n':
                    if pred_value == 'n':
                        TN += 1
                    elif pred_value == 'p':
                        FP += 1
            # except:
            #     """"""
        if TP + FP > 0:
            Prec = TP / (TP + FP)
        else:
            Prec = 0
        if TP + FN > 0:
            Rec = TP / (TP + FN)
        else:
            Rec = 0

        acc = (TP + TN)/(TP + TN + FP + FN)

        return val_loss, Prec, Rec, acc

    def train_triplet_loss(self):
        iterations = 0
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']
        # self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=F.cosine_similarity)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.05)
        for epoch in range(config.train.num_epochs):
            """Make dataloader"""
            train_data = Dataset(config=config, params={'files': train_files_list,
                                                        'mode': 'train', 'triplet': True,
                                                        'files_dict': train})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset(config=config, params={'files': val_files_list,
                                                      'mode': 'train', 'triplet': True,
                                                      'files_dict': val})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                try:
                    anchors = features['anchors']
                    pos = features['pos']
                    neg = features['neg']
                    triplet_files = features['files']
                    self.G = self.G.train()

                    anchor_out = self.G(anchors)
                    pos_out = self.G(pos)
                    neg_out = self.G(neg)


                    loss = self.triplet_loss(anchor_out, pos_out, neg_out)

                    # Backward and optimize.
                    self.reset_grad()
                    loss.backward()
                    self.g_optimizer.step()

                    if iterations % self.log_step == 0:
                        print(str(iterations) + ', loss: ' + str(loss.sum().item()))
                        if self.use_tensorboard:
                            # self.logger.scalar_summary('loss', loss.sum().item(), iterations)
                            self.logger.add_scalar('loss', loss.sum().item(), iterations)

                    if iterations % self.model_save_step == 0:
                        """Calculate validation loss"""
                        val_loss = self.val_loss(val=val_gen, iterations=iterations)
                        print(str(iterations) + ', val_loss: ' + str(val_loss))
                        if self.use_tensorboard:
                            self.logger.add_scalar('val_loss', val_loss, iterations)
                    """Save model checkpoints."""
                    if iterations % self.model_save_step == 0:
                        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                        torch.save({'model': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict()}, G_path)
                        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                    iterations += 1
                except:
                    """"""

    def train_classifier(self):
        iterations = 0
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']
        # self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=F.cosine_similarity)
        self.crossent_loss = nn.CrossEntropyLoss(reduction='none')
        for epoch in range(config.train.num_epochs):
            """Make dataloader"""
            train_data = Dataset(config=config, params={'files': train_files_list,
                                                        'mode': 'train', 'triplet': False,
                                                        'files_dict': train})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset(config=config, params={'files': val_files_list,
                                                      'mode': 'train', 'triplet': False,
                                                      'files_dict': val})
            val_gen = data.DataLoader(val_data, batch_size=1,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                # try:
                    feature = features['features']
                    labels = features['labels']
                    scalers = features['scalers']
                    files = features['files']
                    self.G = self.G.train()

                    predictions = self.G(feature)

                    loss = self.crossent_loss(predictions, labels)
                    """Multiply loss of positive labels by """
                    loss = loss * scalers
                    # Backward and optimize.
                    self.reset_grad()
                    loss.sum().backward()
                    self.g_optimizer.step()

                    if iterations % self.log_step == 0:
                        print(str(iterations) + ', loss: ' + str(loss.sum().item()))
                        if self.use_tensorboard:
                            # self.logger.scalar_summary('loss', loss.sum().item(), iterations)
                            self.logger.add_scalar('loss', loss.sum().item(), iterations)

                    if iterations % self.model_save_step == 0:
                        """Calculate validation loss"""
                        val_loss, Prec, Rec, acc = self.val_loss(val=val_gen, iterations=iterations)
                        print(str(iterations) + ', val_loss: ' + str(val_loss))
                        print(str(iterations) + ', accuracy: ' + str(acc))
                        if self.use_tensorboard:
                            self.logger.add_scalar('val_loss', val_loss, iterations)
                            self.logger.add_scalar('Prec', Prec, iterations)
                            self.logger.add_scalar('Rec', Rec, iterations)
                            self.logger.add_scalar('Accuracy', acc, iterations)
                    """Save model checkpoints."""
                    if iterations % self.model_save_step == 0:
                        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                        torch.save({'model': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict()}, G_path)
                        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                    iterations += 1
                # except:
                #     """"""


    def eval(self):

        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']

        """Make dataloader"""
        """Make dataloader"""
        train_data = Dataset(config=config, params={'files': train_files_list,
                                                    'mode': 'train', 'triplet': True,
                                                    'files_dict': train})
        train_gen = data.DataLoader(train_data, batch_size=1,
                                    shuffle=True, collate_fn=train_data.collate, drop_last=True)
        val_data = Dataset(config=config, params={'files': val_files_list,
                                                  'mode': 'train', 'triplet': True,
                                                  'files_dict': val})
        val_gen = data.DataLoader(val_data, batch_size=1,
                                  shuffle=True, collate_fn=val_data.collate, drop_last=True)
        for part in [train_gen, val_gen]:
            for batch_number, features in tqdm(enumerate(part)):
                try:
                    anchors = features['anchors']
                    pos = features['pos']
                    neg = features['neg']
                    triplet_files = features['files']
                    self.G = self.G.eval()

                    anchor_out = self.G(anchors)
                    anchor_out = np.squeeze(anchor_out.detach().cpu().numpy())

                    file = triplet_files[0]['anchor']  # remember we're using batch size 1
                    metadata = utils.dicova_metadata(file)

                    data_to_save = {'embedding': anchor_out,
                                    'Covid_status': metadata['Covid_status']}
                    dump_path = os.path.join(self.predict_dir, metadata['name'] + '.pkl')
                    joblib.dump(data_to_save, dump_path)
                except:
                    """"""

        """Now you have the predicted embeddings, so predict the label with a variant of KNN"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']
        val_data = Dataset(config=config, params={'files': val_files_list,
                                                  'mode': 'train', 'triplet': True,
                                                  'files_dict': val})
        val_gen = data.DataLoader(val_data, batch_size=1,
                                  shuffle=True, collate_fn=val_data.collate, drop_last=True)
        """Load the embeddings for the entire dataset"""
        files = utils.collect_files(self.predict_dir)
        """Make file to index dictionary"""
        if not os.path.exists(os.path.join(exp_dir, 'feature_matrix.pkl')):
            file2index = {}
            index2file = {}
            feature_matrix = np.zeros(shape=(len(files), config.model.output_dim))
            for i, file in tqdm(enumerate(files)):
                metadata = utils.dicova_metadata(file)
                file2index[metadata['name']] = i
                index2file[i] = metadata['name']
                feat = joblib.load(file)
                feature_matrix[i] = feat['embedding']
            joblib.dump(feature_matrix, os.path.join(exp_dir, 'feature_matrix.pkl'))
            joblib.dump(file2index, os.path.join(exp_dir, 'file2index.pkl'))
            joblib.dump(index2file, os.path.join(exp_dir, 'index2file.pkl'))
        else:
            feature_matrix = joblib.load(os.path.join(exp_dir, 'feature_matrix.pkl'))
            file2index = joblib.load(os.path.join(exp_dir, 'file2index.pkl'))
            index2file = joblib.load(os.path.join(exp_dir, 'index2file.pkl'))
        for batch_number, features in tqdm(enumerate(val_gen)):
            triplet_files = features['files']
            anchor = triplet_files[0]['anchor']
            anchor_path = os.path.join(self.predict_dir, utils.dicova_metadata(anchor)['name'] + '.pkl')
            anchor = joblib.load(anchor_path)
            anchor = anchor['embedding']
            differences = feature_matrix - anchor[None, :]
            distances = np.sum(np.square(differences), axis=1)
            sorted_indices = np.argsort(distances)
            """Let's vote based on top 7 and say if more than 2 are positive, it's positive"""
            sorted_indices = sorted_indices[0:7]
            """Count the votes"""
            positive_votes = 0
            for index in sorted_indices:
                file = index2file[index] + '.pkl'
                covid_status = utils.dicova_metadata(file)
                if covid_status == 'p':
                    positive_votes += 1
            if positive_votes >= 1:
                pred_status = 'p'
            else:
                pred_status = 'n'
            file_data = joblib.load(anchor_path)
            file_data['pred_status'] = pred_status
            joblib.dump(file_data, anchor_path)

    def performance(self):
        """"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']
        val_data = Dataset(config=config, params={'files': val_files_list,
                                                  'mode': 'train', 'triplet': True,
                                                  'files_dict': val})
        val_gen = data.DataLoader(val_data, batch_size=1,
                                  shuffle=True, collate_fn=val_data.collate, drop_last=True)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for batch_number, features in tqdm(enumerate(val_gen)):
            triplet_files = features['files']
            anchor = triplet_files[0]['anchor']
            anchor_path = os.path.join(self.predict_dir, utils.dicova_metadata(anchor)['name'] + '.pkl')
            info = joblib.load(anchor_path)
            if info['Covid_status'] == 'p':
                if info['pred_status'] == 'p':
                    TP += 1
                elif info['pred_status'] == 'n':
                    FN += 1
            elif info['Covid_status'] == 'n':
                if info['pred_status'] == 'n':
                    TN += 1
                elif info['pred_status'] == 'p':
                    FP += 1
        TPR = TP/(TP+FN)
        FPR = FP/(TN+FP)
        stop = None
        """"""

    def to_gpu(self, tensor):
        tensor = tensor.to(self.torch_type)
        tensor = tensor.to(self.device)
        return tensor

    def fix_tensor(self, x):
        x.requires_grad = True
        x = x.to(self.torch_type)
        x = x.cuda()
        return x

    def dump_json(self, dict, path):
        a_file = open(path, "w")
        json.dump(dict, a_file, indent=2)
        a_file.close()

def main():
    solver = Solver()
    if TRAIN:
        solver.train_classifier()
    if EVAL:
        solver.eval()
    if PERFORMANCE:
        solver.performance()


if __name__ == "__main__":
    main()