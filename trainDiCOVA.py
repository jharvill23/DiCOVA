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
from datasetCoswara import Dataset
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

trial = 'trial_1_triplet_loss'
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
        partition = joblib.load(self.partition)
        partition = partition[FOLD]
        train_files = {'positive': partition['train_positive'], 'negative': partition['train_negative']}
        test_files = {'positive': partition['test_positive'], 'negative': partition['test_negative']}
        val_files = {'positive': partition['val_positive'], 'negative': partition['val_negative']}
        return train_files, test_files, val_files

    def val_loss(self, val, iterations):
        val_loss = 0
        for batch_number, features in tqdm(enumerate(val)):
            try:
                spectrograms = features['features']
                files = features['files']
                self.G = self.G.eval()

                outputs, _ = self.G(spectrograms)
                loss = F.mse_loss(spectrograms[:, :, 0:config.data.num_mels],
                                  outputs[:, :, 0:config.data.num_mels], reduction='sum')
                val_loss += loss.item()

                if batch_number == 0:
                    spec = outputs[0]
                    pred_spec = spec.detach().cpu().numpy()
                    true_spec = spectrograms.detach().cpu().numpy()
                    true_spec = true_spec[0]
                    plt.subplot(211)
                    plt.imshow(pred_spec[:, 0:config.data.num_mels].T)
                    plt.subplot(212)
                    plt.imshow(true_spec[:, 0:config.data.num_mels].T)
                    plt.savefig(os.path.join(self.images_dir, str(iterations) + '.png'))
                    plt.close()

            except:
                """"""

        return val_loss

    def train_triplet_loss(self):
        iterations = 0
        """Get train/test"""
        train, test, val = self.get_train_test()
        for epoch in range(config.train.num_epochs):
            """Make dataloader"""


            """*************************YOU ARE HERE**************************"""



            train_data = Dataset(config=config, params={'files': train, 'mode': 'train'})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset(config=config, params={'files': val, 'mode': 'train'})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                try:
                    spectrograms = features['features']
                    files = features['files']
                    self.G = self.G.train()

                    outputs, _ = self.G(spectrograms)
                    loss = F.mse_loss(spectrograms[:, :, 0:config.data.num_mels],
                                      outputs[:, :, 0:config.data.num_mels], reduction='sum')

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

    def eval(self):

        """Get train/test"""
        train, test, val = self.get_train_test()

        """Make dataloader"""
        test_data = Dataset(config=config, params={'files': test, 'mode': 'train'})
        test_gen = data.DataLoader(test_data, batch_size=1,
                                    shuffle=True, collate_fn=test_data.collate, drop_last=True)

        for batch_number, features in tqdm(enumerate(test_gen)):
            try:
                spectrograms = features['features']
                files = features['files']
                self.G = self.G.eval()

                file = files[0]  # remember we're using batch size 1
                metadata = utils.get_metadata(file)

                outputs, bottleneck = self.G(spectrograms)
                bottleneck = np.squeeze(bottleneck.detach().cpu().numpy())

                data_to_save = {'bottleneck': bottleneck,
                                'word': metadata['word'],
                                'ipa_word': metadata['ipa_word']}
                dump_path = os.path.join(self.predict_dir, metadata['filename'] + '.pkl')
                joblib.dump(data_to_save, dump_path)
            except:
                """"""

    def performance(self):
        ft = panphon.FeatureTable()
        dst = panphon.distance.Distance()
        """"""
        if not os.path.exists(os.path.join(exp_dir, 'results_5_words.pkl')):
            vectors = []
            ordered_data = []
            files = collect_files(self.predict_dir)
            for file in tqdm(files):
                data = joblib.load(file)
                ordered_data.append(data)
                vectors.append(data['bottleneck'])
            vectors = np.asarray(vectors)
            results = {}
            distances = 1 - pairwise.cosine_similarity(vectors)
            """Now we want to go through the indices, and get the sorted lists for each utterance"""
            for i, vec in tqdm(enumerate(vectors)):
                if i < 5:
                    filename = utils.get_metadata(files[i])['filename']
                    """We should take the cosine similarity between the vectors"""
                    # difference = np.linalg.norm(vectors - vec[None, :], axis=1)
                    difference = distances[i]
                    ordered_list = np.argsort(difference)
                    """Now just convert ordered list to words"""
                    word_list = []
                    articulatory_difference_list = []
                    for q, element in enumerate(ordered_list):
                        word = ordered_data[element]['word']
                        word_list.append(word)
                        if q == 0:
                            articulatory_difference_list.append(0.0)
                            ground_truth_word = word_list[0]
                            ground_truth_word_ipa = ipa.convert(ground_truth_word)
                        else:
                            pred_word_ipa = ipa.convert(word)
                            dist = dst.dogol_prime_distance(ground_truth_word_ipa, pred_word_ipa)
                            articulatory_difference_list.append(dist)
                    articulatory_difference_list = np.asarray(articulatory_difference_list)
                    arr_reduced = block_reduce(articulatory_difference_list,
                                               block_size=(100,), func=np.mean,
                                               cval=np.mean(articulatory_difference_list))


                    # results[filename] = {'word_list': word_list, 'ordered_list': ordered_list}
                    results[filename] = {'word_list': word_list, 'articulatory_similarity': arr_reduced}
            dump_path = os.path.join(exp_dir, 'results_5_words.pkl')
            joblib.dump(results, dump_path)
        else:
            results = joblib.load(os.path.join(exp_dir, 'results_5_words.pkl'))
            legend = []
            for key, value in results.items():
                plt.plot(value['articulatory_similarity'][:-1])
                legend.append(value['word_list'][0])
            plt.legend(legend)
            plt.show()
            stop = None

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
        solver.train_triplet_loss()
    if EVAL:
        solver.eval()
    if PERFORMANCE:
        solver.performance()


if __name__ == "__main__":
    main()