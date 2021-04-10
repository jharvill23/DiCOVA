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
    TRIAL = args.TRIAL
    if not os.path.isdir(TRIAL):
        os.mkdir(TRIAL)
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

    """Higher layers ablation"""
    high_layers_ablation_models = {
        '1': ['./exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/161000-G.ckpt',
              './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/182500-G.ckpt',
              './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/184500-G.ckpt'],
        '2': ['./exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/97000-G.ckpt',
              './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/82000-G.ckpt',
              './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/173000-G.ckpt'],
        '3': ['./exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/48000-G.ckpt',
              './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/69000-G.ckpt',
              './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/79000-G.ckpt'],
        '4': ['./exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/154000-G.ckpt',
              './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/160500-G.ckpt',
              './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/169500-G.ckpt'],
        '5': ['./exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/33000-G.ckpt',
              './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/47000-G.ckpt',
              './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_high_layers/models/132500-G.ckpt']}

    """SpecAugment 50% ablation"""
    specaug_ablation_models = {
        '1': ['./exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/58000-G.ckpt',
              './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/67000-G.ckpt',
              './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/83500-G.ckpt'],
        '2': ['./exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/102000-G.ckpt',
              './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/122500-G.ckpt',
              './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/118000-G.ckpt'],
        '3': ['./exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/6500-G.ckpt',
              './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/10000-G.ckpt',
              './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/67500-G.ckpt'],
        '4': ['./exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/65000-G.ckpt',
              './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/84000-G.ckpt',
              './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/92000-G.ckpt'],
        '5': ['./exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/97000-G.ckpt',
              './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/29000-G.ckpt',
              './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot5/models/99500-G.ckpt']}

    """SpecAugment 0% ablation"""
    specaug_0_ablation_models = {
        '1': ['./exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/17000-G.ckpt',
              './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/22000-G.ckpt',
              './exps/fold_1_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/24000-G.ckpt'],
        '2': ['./exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/14000-G.ckpt',
              './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/53000-G.ckpt',
              './exps/fold_2_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/62500-G.ckpt'],
        '3': ['./exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/7000-G.ckpt',
              './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/44500-G.ckpt',
              './exps/fold_3_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/69000-G.ckpt'],
        '4': ['./exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/13500-G.ckpt',
              './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/34000-G.ckpt',
              './exps/fold_4_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/79000-G.ckpt'],
        '5': ['./exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/74000-G.ckpt',
              './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/29000-G.ckpt',
              './exps/fold_5_scaling_10_ff_pretraining_coughvid_specaug_prob_0dot0/models/18000-G.ckpt']}

    # for models in [one_ff_ablation_models, high_layers_ablation_models, specaug_ablation_models, best_models]:
    # for models in [specaug_0_ablation_models]:
    #     outfiles = []
    #     if models == one_ff_ablation_models:
    #         name = 'one_ff'
    #     elif models == high_layers_ablation_models:
    #         name = 'high_layers'
    #     elif models == specaug_ablation_models:
    #         name = 'specaug_ablation'
    #     elif models == specaug_0_ablation_models:
    #         name = 'specaug_0_ablation'
    #     elif models == best_models:
    #         name = 'best_models'
    #     for fold in ['1', '2', '3', '4', '5']:
    #         paths = []
    #         eval_paths = []
    #         for checkpoint in models[fold]:
    #             """"""
    #             model_number = checkpoint.split('/')[-1][:-7]
    #             """"""
    #             args.TRIAL = os.path.join(TRIAL, name)
    #             solver = Solver(config=config, training_args=args)
    #             solver.fold = fold
    #             solver.restore_model(G_path=checkpoint)
    #             solver.test_fold = fold
    #             eval_score_path, eval_ensemb_type_dir = solver.eval_ensemble(model_num=model_number)
    #             specific_path, fold_score_path, eval_type_dir = solver.val_scores_ensemble(model_num=model_number)
    #             paths.append(specific_path)
    #             eval_paths.append(eval_score_path)
    #         """We get scores from individual models. Need to load those scores and take mean"""
    #         file_scores = {}
    #         for dictionary in paths:
    #             score_path = dictionary['score_path']
    #             file1 = open(score_path, 'r')
    #             Lines = file1.readlines()
    #             for line in Lines:
    #                 line = line[:-1]
    #                 pieces = line.split(' ')
    #                 filename = pieces[0]
    #                 score = pieces[1]
    #                 if filename not in file_scores:
    #                     file_scores[filename] = [score]
    #                 else:
    #                     file_scores[filename].append(score)
    #         file_final_scores = []
    #         for key, score_list in file_scores.items():
    #             sum = 0
    #             for score in score_list:
    #                 sum += float(score)
    #             sum = sum / len(score_list)
    #             file_final_scores.append(key + ' ' + str(sum))
    #         with open(fold_score_path, 'w') as f:
    #             for item in file_final_scores:
    #                 f.write("%s\n" % item)
    #
    #         eval_file_scores = {}
    #         for x in eval_paths:
    #             score_path = x
    #             file1 = open(score_path, 'r')
    #             Lines = file1.readlines()
    #             for line in Lines:
    #                 line = line[:-1]
    #                 pieces = line.split(' ')
    #                 filename = pieces[0]
    #                 score = pieces[1]
    #                 if filename not in eval_file_scores:
    #                     eval_file_scores[filename] = [score]
    #                 else:
    #                     eval_file_scores[filename].append(score)
    #         eval_file_final_scores = []
    #         for key, score_list in eval_file_scores.items():
    #             sum = 0
    #             for score in score_list:
    #                 sum += float(score)
    #             sum = sum / len(score_list)
    #             eval_file_final_scores.append(key + ' ' + str(sum))
    #         with open(os.path.join(eval_ensemb_type_dir, 'scores'), 'w') as f:
    #             for item in eval_file_final_scores:
    #                 f.write("%s\n" % item)
    #
    #         outfile_path = os.path.join(eval_type_dir, 'outfile.pkl')
    #         utils.scoring(refs=paths[0]['gt_path'], sys_outs=fold_score_path, out_file=outfile_path)
    #         # outfile_path = os.path.join(config.directories.exps, args.TRIAL, 'evaluations', fold, 'val', 'outfile.pkl')
    #         outfiles.append(outfile_path)

    method_outfiles = []
    librispeech_models = []
    for models in [best_models, one_ff_ablation_models, high_layers_ablation_models,
                   specaug_ablation_models, specaug_0_ablation_models, librispeech_models]:
        outfiles = []
        if models == one_ff_ablation_models:
            name = 'one_ff'
        elif models == high_layers_ablation_models:
            name = 'high_layers'
        elif models == specaug_ablation_models:
            name = 'specaug_ablation'
        elif models == specaug_0_ablation_models:
            name = 'specaug_0_ablation'
        elif models == best_models:
            name = 'best_models'
        elif models == librispeech_models:
            name = 'librispeech_ablation'

        """Get the outfiles if they have already been made"""
        outfiles = []
        for fold in ['1', '2', '3', '4', '5']:
            trial = TRIAL
            filepath = os.path.join('./exps', trial, name, 'evaluations', fold, 'val', 'outfile.pkl')
            if os.path.exists(filepath):
                outfiles.append(filepath)

        best_config_dump_path = os.path.join(config.directories.exps, TRIAL, name + '.pkl')
        folder = os.path.join(config.directories.exps, TRIAL)
        get_mean_outfile(outfile_list=outfiles, dump_path=best_config_dump_path)
        method_outfiles.append(best_config_dump_path)



    # outfiles = [best_config_dump_path, linear_regression_dump_path,
    #             random_forest_dump_path, multilayer_perceptron_dump_path]

    names = ['Best Test Config', 'One Future Frame', 'Higher Layers', '50% SpecAugment', '0% SpecAugment', 'LibriSpeech Pretraining']

    utils.eval_summary_paper_plotting(folname=folder, outfiles=method_outfiles, names=names)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to train classifier')
    # parser.add_argument('--TRIAL', type=str, default='Evaluations_scaling_10_ff_pretraining_coughvid_specaug_prob_1dot0_ensemble')
    parser.add_argument('--TRIAL', type=str, default='Evaluations_ablations')
    parser.add_argument('--TRAIN', action='store_true', default=False)
    parser.add_argument('--LOAD_MODEL', action='store_true', default=False)
    parser.add_argument('--FOLD', type=str, default='1')
    args = parser.parse_args()
    main(args)