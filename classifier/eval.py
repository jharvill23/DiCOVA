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
from classifier.train import Solver

config = get_config.get()

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)


def main(args):
    best_models = {'1': './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/71000-G.ckpt',
                   '2': './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/139000-G.ckpt',
                   '3': './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/125000-G.ckpt',
                   '4': './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/72500-G.ckpt',
                   '5': './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/103500-G.ckpt'}
    # outfiles = []
    # for fold in ['1', '2', '3', '4', '5']:
    #     """"""
    #     solver = Solver(config=config, training_args=args)
    #     solver.fold = fold
    #     solver.restore_model(G_path=best_models[fold])
    #     solver.test_fold = fold
    #     solver.eval()
    #     solver.val_scores()
    #     outfile_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'outfile.pkl')
    #     outfiles.append(outfile_path)
    # folder = os.path.join(config.directories.exps, args.TRIAL, 'evaluations')
    # utils.eval_summary(folname=folder, outfiles=outfiles)

    """Take mean of probability for each fold on test data"""
    val_scores = []
    test_scores = {}
    for fold in ['1', '2', '3', '4', '5']:
        val_file = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'scores')
        test_file = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'blind', 'scores')

        file1 = open(val_file, 'r')
        Lines = file1.readlines()
        for line in Lines:
            line = line[:-1]
            pieces = line.split(' ')
            filename = pieces[0]
            score = pieces[1]
            val_scores.append(filename + ' ' + str(score))

        file1 = open(test_file, 'r')
        Lines = file1.readlines()
        for line in Lines:
            line = line[:-1]
            pieces = line.split(' ')
            filename = pieces[0]
            score = pieces[1]
            if filename not in test_scores:
                test_scores[filename] = [score]
            else:
                test_scores[filename].append(score)
    test_final_scores = []
    for key, score_list in test_scores.items():
        sum = 0
        for score in score_list:
            sum += float(score)
        sum = sum/len(score_list)
        test_final_scores.append(key + ' ' + str(sum))
    test_score_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', 'test_scores.txt')
    val_score_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', 'val_scores.txt')
    with open(test_score_path, 'w') as f:
        for item in test_final_scores:
            f.write("%s\n" % item)
    with open(val_score_path, 'w') as f:
        for item in val_scores:
            f.write("%s\n" % item)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    parser.add_argument('--TRIAL', type=str, default='Evaluations_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_ensemble')
    parser.add_argument('--TRAIN', action='store_true', default=False)
    parser.add_argument('--LOAD_MODEL', action='store_true', default=False)
    parser.add_argument('--FOLD', type=str, default='1')
    args = parser.parse_args()
    main(args)