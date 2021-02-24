import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import model
import shutil
from utils import collect_files
import utils
from torch.utils import data
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from config import get_config
from dataSetCode import dicova
from dataSetCode.dicova import Dataset
import torch.nn.functional as F


config = get_config.get()

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'finetuning_trial_1'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

FOLD = '1'
TRAIN = True
LOAD_MODEL = False

class Solver(object):
    """Solver"""

    def __init__(self, config):
        """Initialize configurations."""

        self.config = config

        # Training configurations.
        self.g_lr = self.config.classifier.lr
        self.torch_type = torch.float32

        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = os.path.join(exp_dir, 'logs')
        self.model_save_dir = os.path.join(exp_dir, 'models')
        self.train_data_dir = self.config.directories.features
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

        # if not os.path.exists('dicova_partitions.pkl'):
        #     utils.get_dicova_partitions()

        """Training Data"""
        self.training_data = dicova.DiCOVA(config=self.config)

        """Partition file"""
        if TRAIN:
            # copy config
            shutil.copy(src=get_config.path(), dst=os.path.join(exp_dir, 'config.yml'))

        # Step size.
        self.log_step = self.config.train.log_step
        self.model_save_step = self.config.train.model_save_step

        # Build the model
        self.build_model()
        if LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        self.G = model.CTCmodel(self.config)
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
        G_path = None
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
        self.training_data.feature_path_partition()
        partition = self.training_data.feat_partition
        partition = partition[FOLD]

        train_files = {'positive': partition['train_pos'], 'negative': partition['train_neg']}
        # test_files = {'positive': partition['test_positive'], 'negative': partition['test_negative']}
        val_files = {'positive': partition['val_pos'], 'negative': partition['val_neg']}
        return train_files, val_files

    def val_loss(self, val, iterations):
        val_loss = 0
        for batch_number, features in tqdm(enumerate(val)):
            try:
                feature = features['features']
                files = features['files']
                self.G = self.G.eval()
                predictions = self.G(feature)
                input_length = feature.shape[1]
                """Now we need to trim the input features for MSE error between output"""
                feature = feature[:, self.config.pretraining.future_frames:, :]
                predictions = predictions[:, 0: input_length - self.config.pretraining.future_frames, :]
                loss = F.mse_loss(input=feature, target=predictions)
                val_loss += loss.item()
            except:
                """"""
        return val_loss

    def train(self):
        iterations = 0
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']
        for epoch in range(self.config.train.num_epochs):
            """Make dataloader"""


            """*************************** YOU ARE HERE *****************************"""


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
                # try:
                    feature = features['features']
                    files = features['files']
                    self.G = self.G.train()
                    predictions = self.G(feature)
                    input_length = feature.shape[1]
                    """Now we need to trim the input features for MSE error between output"""
                    feature = feature[:, self.config.pretraining.future_frames:, :]
                    predictions = predictions[:, 0: input_length-self.config.pretraining.future_frames, :]
                    loss = F.mse_loss(input=feature, target=predictions)
                    # Backward and optimize.
                    self.reset_grad()
                    self.g_optimizer.step()

                    if iterations % self.log_step == 0:
                        normalized_loss = loss.item()/input_length
                        print(str(iterations) + ', loss: ' + str(normalized_loss))
                        if self.use_tensorboard:
                            self.logger.add_scalar('loss', normalized_loss, iterations)

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
                # except:
                #     """"""

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
    solver = Solver(config=config)
    if TRAIN:
        solver.train()


if __name__ == "__main__":
    main()