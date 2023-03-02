import os
import numpy as np
import json
import torch
# from scipy.misc import imread, imresize
from collections import Counter
from random import seed, choice, sample
import random
import pandas as pd


def save_predic(which_fold, gt_dic, pred_dic, rc):
    os.makedirs('./prediction/fold{}'.format(which_fold))
    file_gt_name = './prediction/fold{}/gt_dict.npy'.format(which_fold)
    np.save(file_gt_name, gt_dic)
    file_pred_name = './prediction/fold{}/pred_dict_rc_{}.npy'.format(which_fold, rc)
    np.save(file_pred_name, pred_dic)

class AverageMeter(object):

    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def spearman_correlation(pred_scores, GT_scores):

    assert len(pred_scores) == len(GT_scores)

    evaluation_data = pd.DataFrame({
        'pred': pred_scores,
        'GT': GT_scores
    })
    correlation = evaluation_data.corr('spearman').values[0, 1]

    return correlation


def test_correlation():

    pred_scores = list(range(0, 2000))
    GT_scores = list(range(2000, 4000))
    spearman_score = spearman_correlation(pred_scores, GT_scores)
    print(spearman_score)


class TimeShow(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.day = 0
        self.hour = 0
        self.min = 0
        self.sec = 0

    def update(self, time):
        self.day = int(time // (3600*24))
        self.hour = int((time // 3600) % 24)
        self.min = int((time // 60) % 60)
        self.sec = int(time % 60)


if __name__ == '__main__':
    test_correlation()


