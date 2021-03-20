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

def get_mean_outfile(outfile_list, dump_path):
    """Let's extract relevant data ourselves"""
    R = []
    for file in outfile_list:
        res = joblib.load(file)
        R.append(res)
    data_x, data_y, data_auc = [], [], []
    for i in range(len(R)):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
    data_x = np.mean(np.array(data_x), axis=0)
    data_y = np.mean(np.array(data_y), axis=0)
    data_auc = np.mean(np.array(data_auc))

def main(args):
    """Best performing set"""
    best_models = []
    best_models = {'1': ['./exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/71000-G.ckpt',
                         './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/84000-G.ckpt',
                         './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/156500-G.ckpt'],
                   '2': ['./exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/139000-G.ckpt',
                         './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/117500-G.ckpt',
                         './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/110000-G.ckpt'],
                   '3': ['./exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/125000-G.ckpt',
                         './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/123000-G.ckpt',
                         './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/85500-G.ckpt'],
                   '4': ['./exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/72500-G.ckpt',
                         './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/36000-G.ckpt',
                         './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/60000-G.ckpt'],
                   '5': ['./exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/103500-G.ckpt',
                         './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/47500-G.ckpt',
                         './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0/models/65000-G.ckpt']}

    outfiles = []
    for fold in ['1', '2', '3', '4', '5']:
        paths = []
        eval_paths = []
        for checkpoint in best_models[fold]:
            """"""
            model_number = checkpoint.split('/')[-1][:-7]
            """"""
            solver = Solver(config=config, training_args=args)
            solver.fold = fold
            solver.restore_model(G_path=checkpoint)
            solver.test_fold = fold
            eval_score_path, eval_ensemb_type_dir = solver.eval_ensemble(model_num=model_number)
            specific_path, fold_score_path, eval_type_dir = solver.val_scores_ensemble(model_num=model_number)
            paths.append(specific_path)
            eval_paths.append(eval_score_path)
        """We get scores from individual models. Need to load those scores and take mean"""
        file_scores = {}
        for dictionary in paths:
            score_path = dictionary['score_path']
            file1 = open(score_path, 'r')
            Lines = file1.readlines()
            for line in Lines:
                line = line[:-1]
                pieces = line.split(' ')
                filename = pieces[0]
                score = pieces[1]
                if filename not in file_scores:
                    file_scores[filename] = [score]
                else:
                    file_scores[filename].append(score)
        file_final_scores = []
        for key, score_list in file_scores.items():
            sum = 0
            for score in score_list:
                sum += float(score)
            sum = sum / len(score_list)
            file_final_scores.append(key + ' ' + str(sum))
        with open(fold_score_path, 'w') as f:
            for item in file_final_scores:
                f.write("%s\n" % item)

        eval_file_scores = {}
        for x in eval_paths:
            score_path = x
            file1 = open(score_path, 'r')
            Lines = file1.readlines()
            for line in Lines:
                line = line[:-1]
                pieces = line.split(' ')
                filename = pieces[0]
                score = pieces[1]
                if filename not in eval_file_scores:
                    eval_file_scores[filename] = [score]
                else:
                    eval_file_scores[filename].append(score)
        eval_file_final_scores = []
        for key, score_list in eval_file_scores.items():
            sum = 0
            for score in score_list:
                sum += float(score)
            sum = sum / len(score_list)
            eval_file_final_scores.append(key + ' ' + str(sum))
        with open(os.path.join(eval_ensemb_type_dir, 'scores'), 'w') as f:
            for item in eval_file_final_scores:
                f.write("%s\n" % item)

        outfile_path = os.path.join(eval_type_dir, 'outfile.pkl')
        utils.scoring(refs=paths[0]['gt_path'], sys_outs=fold_score_path, out_file=outfile_path)
        # outfile_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'outfile.pkl')
        outfiles.append(outfile_path)

    # """Get the outfiles if they have already been made"""
    # outfiles = []
    # for fold in ['1', '2', '3', '4', '5']:
    #     trial = args.TRIAL
    #     filepath = os.path.join('./exps', trial, 'evaluations', fold, 'val', 'outfile.pkl')
    #     if os.path.exists(filepath):
    #         outfiles.append(filepath)
    #
    # # folder = os.path.join(config.directories.exps, args.TRIAL, 'evaluations')
    # get_mean_outfile(outfile_list=outfiles)
    # stop = None

    # utils.eval_summary(folname=folder, outfiles=outfiles)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    # parser.add_argument('--TRIAL', type=str, default='Evaluations_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_ensemble')
    parser.add_argument('--TRIAL', type=str, default='Evaluations_best_config_vs_baselines')
    parser.add_argument('--TRAIN', action='store_true', default=False)
    parser.add_argument('--LOAD_MODEL', action='store_true', default=False)
    parser.add_argument('--FOLD', type=str, default='1')
    args = parser.parse_args()
    main(args)