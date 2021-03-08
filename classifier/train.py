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
import copy
from scipy.special import softmax
import argparse


config = get_config.get()

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

# FOLD = '1'
# TRAIN = True
# LOAD_MODEL = False

class Solver(object):
    """Solver"""

    def __init__(self, config, training_args):
        """Initialize configurations."""

        self.config = config
        self.fold = training_args.FOLD
        self.val_folds = os.path.join('val_folds', 'fold_' + self.fold, 'val_labels')

        # Training configurations.
        self.g_lr = self.config.post_pretraining_classifier.lr
        self.torch_type = torch.float32

        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        # trial = 'finetuning_trial_10_with_scaling_and_auc_plots_10_ff_pretraining_coughvid_specaug_prob_0dot7'
        trial = training_args.TRIAL
        self.exp_dir = os.path.join(self.config.directories.exps, trial)
        if not os.path.isdir(self.exp_dir):
            os.mkdir(self.exp_dir)
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.model_save_dir = os.path.join(self.exp_dir, 'models')
        self.train_data_dir = self.config.directories.features
        self.predict_dir = os.path.join(self.exp_dir, 'predictions')
        self.images_dir = os.path.join(self.exp_dir, 'images')
        self.val_scores_dir = os.path.join(self.exp_dir, 'val_scores')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.predict_dir):
            os.mkdir(self.predict_dir)
        if not os.path.isdir(self.images_dir):
            os.mkdir(self.images_dir)
        if not os.path.isdir(self.val_scores_dir):
            os.mkdir(self.val_scores_dir)


        # if not os.path.exists('dicova_partitions.pkl'):
        #     utils.get_dicova_partitions()

        """Training Data"""
        self.training_data = dicova.DiCOVA(config=self.config)

        """Partition file"""
        if training_args.TRAIN:
            # copy config
            shutil.copy(src=get_config.path(), dst=os.path.join(self.exp_dir, 'config.yml'))

        # Step size.
        self.log_step = self.config.train.log_step
        self.model_save_step = self.config.train.model_save_step

        # Build the model
        self.build_model()
        if training_args.LOAD_MODEL:
            self.restore_model(training_args.RESTORE_PATH)
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        pretrain_config = copy.deepcopy(self.config)
        pretrain_config.model.name = 'PreTrainer2'
        """Load the weights"""
        self.pretrained = model.CTCmodel(pretrain_config)
        # pretrain_checkpoint = self._load('./exps/pretraining_trial_2/models/20000-G.ckpt')
        # pretrain_checkpoint = self._load('./exps/pretraining_trial_3_10_future_frames/models/140000-G.ckpt')
        pretrain_checkpoint = self._load('./exps/pretraining_trial_4_coughvid_10_future_frames/models/19500-G.ckpt')
        self.pretrained.load_state_dict(pretrain_checkpoint['model'])
        """Freeze pretrainer"""
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.pretrained.to(self.device)
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

    def restore_model(self, G_path):
        """Restore the model"""
        print('Loading the trained models... ')
        # G_path = None
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
        partition = partition[self.fold]

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
        ground_truth = []
        pred_scores = []
        for batch_number, features in tqdm(enumerate(val)):
            try:
                feature = features['features']
                files = features['files']
                labels = features['labels']
                scalers = features['scalers']
                self.G = self.G.eval()
                _, intermediate = self.pretrained(feature)
                predictions = self.G(intermediate)
                loss = self.crossent_loss(predictions, labels)
                """Multiply loss of positive labels by """
                loss = loss * scalers
                val_loss += loss.sum().item()

                predictions = np.squeeze(predictions.detach().cpu().numpy())
                max_preds = np.argmax(predictions, axis=1)
                scores = softmax(predictions, axis=1)
                pred_value = [self.index2class[x] for x in max_preds]

                info = [self.training_data.get_file_metadata(x) for x in files]

                for i, file in enumerate(files):
                    filekey = file.split('/')[-1][:-4]
                    gt = info[i]['Covid_status']
                    score = scores[i, self.class2index['p']]
                    ground_truth.append(filekey + ' ' + gt)
                    pred_scores.append(filekey + ' ' + str(score))

                for i, entry in enumerate(info):
                    if entry['Covid_status'] == 'p':
                        if pred_value[i] == 'p':
                            TP += 1
                        elif pred_value[i] == 'n':
                            FN += 1
                    elif entry['Covid_status'] == 'n':
                        if pred_value[i] == 'n':
                            TN += 1
                        elif pred_value[i] == 'p':
                            FP += 1

            except:
                """"""
        """Sort the lists in alphabetical order"""
        ground_truth.sort()
        pred_scores.sort()

        """Write the files"""
        gt_path = os.path.join(self.val_scores_dir, 'val_labels_' + str(iterations))
        score_path = os.path.join(self.val_scores_dir, 'scores_' + str(iterations))

        for path in [gt_path, score_path]:
            with open(path, 'w') as f:
                if path == gt_path:
                    for item in ground_truth:
                        f.write("%s\n" % item)
                elif path == score_path:
                    for item in pred_scores:
                        f.write("%s\n" % item)
        out_file_path = os.path.join(self.val_scores_dir, 'outfile_' + str(iterations) + '.pkl')
        utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)
        auc = utils.summary(folname=self.val_scores_dir, scores=out_file_path, iterations=iterations)

        if TP + FP > 0:
            Prec = TP / (TP + FP)
        else:
            Prec = 0
        if TP + FN > 0:
            Rec = TP / (TP + FN)
        else:
            Rec = 0

        acc = (TP + TN) / (TP + TN + FP + FN)

        return val_loss, Prec, Rec, acc, auc

    def train(self):
        iterations = 0
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']
        self.crossent_loss = nn.CrossEntropyLoss(reduction='none')
        for epoch in range(self.config.train.num_epochs):
            """Make dataloader"""
            train_data = Dataset(config=config, params={'files': train_files_list,
                                                        'mode': 'train',
                                                        'data_object': self.training_data,
                                                        'specaugment': self.config.train.specaugment})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            self.index2class = train_data.index2class
            self.class2index = train_data.class2index
            val_data = Dataset(config=config, params={'files': val_files_list,
                                                      'mode': 'train',
                                                      'data_object': self.training_data,
                                                      'specaugment': False})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                      shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                try:
                    feature = features['features']
                    files = features['files']
                    labels = features['labels']
                    scalers = features['scalers']
                    self.G = self.G.train()
                    _, intermediate = self.pretrained(feature)
                    predictions = self.G(intermediate)
                    loss = self.crossent_loss(predictions, labels)
                    """Multiply loss of positive labels by """
                    loss = loss * scalers
                    # Backward and optimize.
                    self.reset_grad()
                    loss.sum().backward()
                    self.g_optimizer.step()

                    if iterations % self.log_step == 0:
                        normalized_loss = loss.sum().item()
                        print(str(iterations) + ', loss: ' + str(normalized_loss))
                        if self.use_tensorboard:
                            self.logger.add_scalar('loss', normalized_loss, iterations)

                    if iterations % self.model_save_step == 0:
                        """Calculate validation loss"""
                        val_loss, Prec, Rec, acc, auc = self.val_loss(val=val_gen, iterations=iterations)
                        print(str(iterations) + ', val_loss: ' + str(val_loss))
                        if self.use_tensorboard:
                            self.logger.add_scalar('val_loss', val_loss, iterations)
                            self.logger.add_scalar('Prec', Prec, iterations)
                            self.logger.add_scalar('Rec', Rec, iterations)
                            self.logger.add_scalar('Accuracy', acc, iterations)
                            self.logger.add_scalar('AUC', auc, iterations)
                    """Save model checkpoints."""
                    if iterations % self.model_save_step == 0:
                        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                        torch.save({'model': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict()}, G_path)
                        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                    iterations += 1
                except:
                    """"""

    def val_scores(self):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'val')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']

        ground_truth = []
        pred_scores = []

        """Make dataloader"""
        train_data = Dataset(config=config, params={'files': train_files_list,
                                                    'mode': 'train',
                                                    'data_object': self.training_data,
                                                    'specaugment': self.config.train.specaugment})
        train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                    shuffle=True, collate_fn=train_data.collate, drop_last=True)
        self.index2class = train_data.index2class
        self.class2index = train_data.class2index
        val_data = Dataset(config=config, params={'files': val_files_list,
                                                  'mode': 'train',
                                                  'data_object': self.training_data,
                                                  'specaugment': False})
        val_gen = data.DataLoader(val_data, batch_size=1, shuffle=True, collate_fn=val_data.collate, drop_last=False)
        for batch_number, features in tqdm(enumerate(val_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            info = [self.training_data.get_file_metadata(x) for x in files]
            file = files[0]  # batch size 1

            filekey = file.split('/')[-1][:-4]
            gt = info[0]['Covid_status']
            score = scores[0, self.class2index['p']]
            ground_truth.append(filekey + ' ' + gt)
            pred_scores.append(filekey + ' ' + str(score))
        """Sort the lists in alphabetical order"""
        ground_truth.sort()
        pred_scores.sort()

        """Write the files"""
        gt_path = os.path.join(self.eval_type_dir, 'val_labels')
        score_path = os.path.join(self.eval_type_dir, 'scores')
        for path in [gt_path, score_path]:
            with open(path, 'w') as f:
                if path == gt_path:
                    for item in ground_truth:
                        f.write("%s\n" % item)
                elif path == score_path:
                    for item in pred_scores:
                        f.write("%s\n" % item)
        out_file_path = os.path.join(self.eval_type_dir, 'outfile.pkl')
        utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)

    def eval(self):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'blind')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        """Get test files"""
        test_files = utils.collect_files(self.config.directories.dicova_test_logspect_feats)
        """Make dataloader"""
        test_data = Dataset(config=config, params={'files': test_files,
                                                    'mode': 'test',
                                                    'data_object': None,
                                                    'specaugment': 0.0})
        test_gen = data.DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=test_data.collate, drop_last=False)
        self.index2class = test_data.index2class
        self.class2index = test_data.class2index
        pred_scores = []
        for batch_number, features in tqdm(enumerate(test_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)

            file = files[0]  # batch size is 1 for evaluation
            filekey = file.split('/')[-1][:-4]
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            score = scores[0, self.class2index['p']]
            pred_scores.append(filekey + ' ' + str(score))
        pred_scores.sort()
        score_path = os.path.join(self.eval_type_dir, 'scores')
        with open(score_path, 'w') as f:
            for item in pred_scores:
                f.write("%s\n" % item)

    def val_scores_ensemble(self, model_num):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'val')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        self.specific_model_dir = os.path.join(self.eval_type_dir, model_num)
        if not os.path.isdir(self.specific_model_dir):
            os.mkdir(self.specific_model_dir)
        """Get train/test"""
        train, val = self.get_train_test()
        train_files_list = train['positive'] + train['negative']
        val_files_list = val['positive'] + val['negative']

        ground_truth = []
        pred_scores = []

        """Make dataloader"""
        train_data = Dataset(config=config, params={'files': train_files_list,
                                                    'mode': 'train',
                                                    'data_object': self.training_data,
                                                    'specaugment': self.config.train.specaugment})
        train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                    shuffle=True, collate_fn=train_data.collate, drop_last=True)
        self.index2class = train_data.index2class
        self.class2index = train_data.class2index
        val_data = Dataset(config=config, params={'files': val_files_list,
                                                  'mode': 'train',
                                                  'data_object': self.training_data,
                                                  'specaugment': False})
        val_gen = data.DataLoader(val_data, batch_size=1, shuffle=True, collate_fn=val_data.collate, drop_last=False)
        for batch_number, features in tqdm(enumerate(val_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            info = [self.training_data.get_file_metadata(x) for x in files]
            file = files[0]  # batch size 1

            filekey = file.split('/')[-1][:-4]
            gt = info[0]['Covid_status']
            score = scores[0, self.class2index['p']]
            ground_truth.append(filekey + ' ' + gt)
            pred_scores.append(filekey + ' ' + str(score))
        """Sort the lists in alphabetical order"""
        ground_truth.sort()
        pred_scores.sort()

        """Write the files"""
        gt_path = os.path.join(self.specific_model_dir, 'val_labels')
        score_path = os.path.join(self.specific_model_dir, 'scores')
        for path in [gt_path, score_path]:
            with open(path, 'w') as f:
                if path == gt_path:
                    for item in ground_truth:
                        f.write("%s\n" % item)
                elif path == score_path:
                    for item in pred_scores:
                        f.write("%s\n" % item)
        paths = {'gt_path': gt_path, 'score_path': score_path}
        return paths, os.path.join(self.eval_type_dir, 'scores'), self.eval_type_dir
        # out_file_path = os.path.join(self.specific_model_dir, 'outfile.pkl')
        # utils.scoring(refs=gt_path, sys_outs=score_path, out_file=out_file_path)

    def eval_ensemble(self, model_num):
        self.evaluation_dir = os.path.join(self.exp_dir, 'evaluations')
        if not os.path.exists(self.evaluation_dir):
            os.mkdir(self.evaluation_dir)
        self.fold_dir = os.path.join(self.evaluation_dir, self.test_fold)
        if not os.path.isdir(self.fold_dir):
            os.mkdir(self.fold_dir)
        self.eval_type_dir = os.path.join(self.fold_dir, 'blind')
        if not os.path.isdir(self.eval_type_dir):
            os.mkdir(self.eval_type_dir)
        self.specific_model_dir = os.path.join(self.eval_type_dir, model_num)
        if not os.path.isdir(self.specific_model_dir):
            os.mkdir(self.specific_model_dir)
        """Get test files"""
        test_files = utils.collect_files(self.config.directories.dicova_test_logspect_feats)
        """Make dataloader"""
        test_data = Dataset(config=config, params={'files': test_files,
                                                    'mode': 'test',
                                                    'data_object': None,
                                                    'specaugment': 0.0})
        test_gen = data.DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=test_data.collate, drop_last=False)
        self.index2class = test_data.index2class
        self.class2index = test_data.class2index
        pred_scores = []
        for batch_number, features in tqdm(enumerate(test_gen)):
            feature = features['features']
            files = features['files']
            self.G = self.G.eval()
            _, intermediate = self.pretrained(feature)
            predictions = self.G(intermediate)

            file = files[0]  # batch size is 1 for evaluation
            filekey = file.split('/')[-1][:-4]
            predictions = predictions.detach().cpu().numpy()
            scores = softmax(predictions, axis=1)
            score = scores[0, self.class2index['p']]
            pred_scores.append(filekey + ' ' + str(score))
        pred_scores.sort()
        score_path = os.path.join(self.specific_model_dir, 'scores')
        with open(score_path, 'w') as f:
            for item in pred_scores:
                f.write("%s\n" % item)
        return score_path, self.eval_type_dir

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

def main(args):
    solver = Solver(config=config, training_args=args)
    if args.TRAIN:
        solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    parser.add_argument('--TRIAL', type=str, default='dummy_exp')
    parser.add_argument('--TRAIN', action='store_true', default=True)
    parser.add_argument('--LOAD_MODEL', action='store_true', default=False)
    parser.add_argument('--FOLD', type=str, default='1')
    parser.add_argument('--RESTORE_PATH', type=str, default='')
    args = parser.parse_args()
    main(args)