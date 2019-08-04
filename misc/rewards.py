from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable

import sys
sys.path.append("cider")
import os
sys.path.append(
    os.path.join(os.path.dirname(__file__), os.path.pardir, 'cider'))
from pyciderevalcap.ciderD.ciderD import CiderD

CiderD_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(data, gen_result, greedy_res,
                             return_gen_scores=False):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(
            data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    print('Cider scores:', _)
    scores = cider_scores
    
    cider_gen = scores[:batch_size]
    print('batch size is {}, cider rewards mean is {} \n'.format(
        batch_size, cider_gen.mean()))

    cider_greedy = scores[batch_size:].mean()
    
    scores = scores[:batch_size] - scores[batch_size:]

    # rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    if (not return_gen_scores):
        return scores, cider_greedy
    else:
        return cider_gen, scores, cider_greedy