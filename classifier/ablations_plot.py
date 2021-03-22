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
import pickle

config = get_config.get()

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

def get_mean_outfile(outfile_list, dump_path):
    """Let's extract relevant data ourselves"""
    R = []
    for file in outfile_list:
        res = joblib.load(file)
        R.append(res)
    TPR, FPR, AUC, sensitivity, specificity, thresholds = [], [], [], [], [], []
    for i in range(len(R)):
        TPR.append(R[i]['TPR'])
        FPR.append(R[i]['FPR'])
        AUC.append(R[i]['AUC'])
        sensitivity.append(R[i]['sensitivity'])
        specificity.append(R[i]['specificity'])
        thresholds.append(R[i]['thresholds'])
    TPR = np.mean(np.asarray(TPR), axis=0)
    FPR = np.mean(np.asarray(FPR), axis=0)
    AUC = np.mean(np.asarray(AUC), axis=0)
    sensitivity = np.mean(np.asarray(sensitivity), axis=0)
    specificity = np.mean(np.asarray(specificity), axis=0)
    thresholds = np.mean(np.asarray(thresholds), axis=0)

    scores = {'TPR': TPR,
              'FPR': FPR,
              'AUC': AUC,
              'sensitivity': sensitivity,
              'specificity': specificity,
              'thresholds': thresholds}
    with open(dump_path, "wb") as f:
        pickle.dump(scores, f)

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

    """1 future frame ablation
    REMEMBER TO CHANGE CONFIG FILE TO HAVE 1ff when you run these models"""
    one_ff_ablation_models = {'1': ['./exps/fold_1_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/38500-G.ckpt',
                                    './exps/fold_1_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/20000-G.ckpt',
                                    './exps/fold_1_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/65000-G.ckpt'],
                              '2': ['./exps/fold_2_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/73500-G.ckpt',
                                    './exps/fold_2_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/146500-G.ckpt',
                                    './exps/fold_2_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/172000-G.ckpt'],
                              '3': ['./exps/fold_3_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/37500-G.ckpt',
                                    './exps/fold_3_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/132500-G.ckpt',
                                    './exps/fold_3_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/183500-G.ckpt'],
                              '4': ['./exps/fold_4_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/10000-G.ckpt',
                                    './exps/fold_4_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/20000-G.ckpt',
                                    './exps/fold_4_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/87500-G.ckpt'],
                              '5': ['./exps/fold_5_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/47500-G.ckpt',
                                    './exps/fold_5_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/14500-G.ckpt',
                                    './exps/fold_5_scaling_1_ff_pretraining_coughvid_specaug_prob_1dot0/models/152000-G.ckpt']}

    # outfiles = []
    # for fold in ['1', '2', '3', '4', '5']:
    #     paths = []
    #     eval_paths = []
    #     for checkpoint in best_models[fold]:
    #         """"""
    #         model_number = checkpoint.split('/')[-1][:-7]
    #         """"""
    #         solver = Solver(config=config, training_args=args)
    #         solver.fold = fold
    #         solver.restore_model(G_path=checkpoint)
    #         solver.test_fold = fold
    #         eval_score_path, eval_ensemb_type_dir = solver.eval_ensemble(model_num=model_number)
    #         specific_path, fold_score_path, eval_type_dir = solver.val_scores_ensemble(model_num=model_number)
    #         paths.append(specific_path)
    #         eval_paths.append(eval_score_path)
    #     """We get scores from individual models. Need to load those scores and take mean"""
    #     file_scores = {}
    #     for dictionary in paths:
    #         score_path = dictionary['score_path']
    #         file1 = open(score_path, 'r')
    #         Lines = file1.readlines()
    #         for line in Lines:
    #             line = line[:-1]
    #             pieces = line.split(' ')
    #             filename = pieces[0]
    #             score = pieces[1]
    #             if filename not in file_scores:
    #                 file_scores[filename] = [score]
    #             else:
    #                 file_scores[filename].append(score)
    #     file_final_scores = []
    #     for key, score_list in file_scores.items():
    #         sum = 0
    #         for score in score_list:
    #             sum += float(score)
    #         sum = sum / len(score_list)
    #         file_final_scores.append(key + ' ' + str(sum))
    #     with open(fold_score_path, 'w') as f:
    #         for item in file_final_scores:
    #             f.write("%s\n" % item)
    #
    #     eval_file_scores = {}
    #     for x in eval_paths:
    #         score_path = x
    #         file1 = open(score_path, 'r')
    #         Lines = file1.readlines()
    #         for line in Lines:
    #             line = line[:-1]
    #             pieces = line.split(' ')
    #             filename = pieces[0]
    #             score = pieces[1]
    #             if filename not in eval_file_scores:
    #                 eval_file_scores[filename] = [score]
    #             else:
    #                 eval_file_scores[filename].append(score)
    #     eval_file_final_scores = []
    #     for key, score_list in eval_file_scores.items():
    #         sum = 0
    #         for score in score_list:
    #             sum += float(score)
    #         sum = sum / len(score_list)
    #         eval_file_final_scores.append(key + ' ' + str(sum))
    #     with open(os.path.join(eval_ensemb_type_dir, 'scores'), 'w') as f:
    #         for item in eval_file_final_scores:
    #             f.write("%s\n" % item)
    #
    #     outfile_path = os.path.join(eval_type_dir, 'outfile.pkl')
    #     utils.scoring(refs=paths[0]['gt_path'], sys_outs=fold_score_path, out_file=outfile_path)
    #     # outfile_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'outfile.pkl')
    #     outfiles.append(outfile_path)

    """Get the outfiles if they have already been made"""
    outfiles = []
    for fold in ['1', '2', '3', '4', '5']:
        trial = args.TRIAL
        filepath = os.path.join('./exps', trial, 'evaluations', fold, 'val', 'outfile.pkl')
        if os.path.exists(filepath):
            outfiles.append(filepath)

    best_config_dump_path = os.path.join(config.directories.exps, args.TRIAL, 'best_config.pkl')
    folder = os.path.join(config.directories.exps, args.TRIAL)
    get_mean_outfile(outfile_list=outfiles, dump_path=best_config_dump_path)

    """Now get the linear regression outfiles"""
    lr_folder = os.path.join(folder, 'linear_regression')
    if not os.path.isdir(lr_folder):
        os.mkdir(lr_folder)
    outfiles = []
    for fold in ['1', '2', '3', '4', '5']:
        score_filepath = '/home/john/Documents/School/Spring_2021/DICOVA/DiCOVA_baseline/results_lr/fold_' + fold + '/val_scores.txt'
        gt_path = '/home/john/Documents/School/Spring_2021/DiCOVA/exps/Evaluations_best_config_vs_baselines/val_labels_per_fold/val_labels_fold_' + fold
        outfile_path = os.path.join(lr_folder, 'fold_' + fold + '.pkl')
        outfiles.append(outfile_path)
        utils.scoring_for_paper(refs=gt_path, sys_outs=score_filepath, out_file=outfile_path)
    linear_regression_dump_path = os.path.join(config.directories.exps, args.TRIAL, 'linear_regression.pkl')
    get_mean_outfile(outfile_list=outfiles, dump_path=linear_regression_dump_path)

    """Now get the random forest outfiles"""
    rf_folder = os.path.join(folder, 'random_forest')
    if not os.path.isdir(rf_folder):
        os.mkdir(rf_folder)
    outfiles = []
    for fold in ['1', '2', '3', '4', '5']:
        score_filepath = '/home/john/Documents/School/Spring_2021/DICOVA/DiCOVA_baseline/results_rf/fold_' + fold + '/val_scores.txt'
        gt_path = '/home/john/Documents/School/Spring_2021/DiCOVA/exps/Evaluations_best_config_vs_baselines/val_labels_per_fold/val_labels_fold_' + fold
        outfile_path = os.path.join(rf_folder, 'fold_' + fold + '.pkl')
        outfiles.append(outfile_path)
        utils.scoring_for_paper(refs=gt_path, sys_outs=score_filepath, out_file=outfile_path)
    random_forest_dump_path = os.path.join(config.directories.exps, args.TRIAL, 'random_forest.pkl')
    get_mean_outfile(outfile_list=outfiles, dump_path=random_forest_dump_path)

    """Now get the multilayer perceptron outfiles"""
    mlp_folder = os.path.join(folder, 'multilayer_perceptron')
    if not os.path.isdir(mlp_folder):
        os.mkdir(mlp_folder)
    outfiles = []
    for fold in ['1', '2', '3', '4', '5']:
        score_filepath = '/home/john/Documents/School/Spring_2021/DICOVA/DiCOVA_baseline/results_mlp/fold_' + fold + '/val_scores.txt'
        gt_path = '/home/john/Documents/School/Spring_2021/DiCOVA/exps/Evaluations_best_config_vs_baselines/val_labels_per_fold/val_labels_fold_' + fold
        outfile_path = os.path.join(mlp_folder, 'fold_' + fold + '.pkl')
        outfiles.append(outfile_path)
        utils.scoring_for_paper(refs=gt_path, sys_outs=score_filepath, out_file=outfile_path)
    multilayer_perceptron_dump_path = os.path.join(config.directories.exps, args.TRIAL, 'multilayer_perceptron.pkl')
    get_mean_outfile(outfile_list=outfiles, dump_path=multilayer_perceptron_dump_path)

    outfiles = [best_config_dump_path, linear_regression_dump_path,
                random_forest_dump_path, multilayer_perceptron_dump_path]

    names = ['Best config', 'Linear Regression', 'Random Forest', 'Multi-layer Perceptron']

    utils.eval_summary_paper_plotting(folname=folder, outfiles=outfiles, names=names)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    # parser.add_argument('--TRIAL', type=str, default='Evaluations_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_ensemble')
    parser.add_argument('--TRIAL', type=str, default='Evaluations_ablations')
    parser.add_argument('--TRAIN', action='store_true', default=False)
    parser.add_argument('--LOAD_MODEL', action='store_true', default=False)
    parser.add_argument('--FOLD', type=str, default='1')
    args = parser.parse_args()
    main(args)