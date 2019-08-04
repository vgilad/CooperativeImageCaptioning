from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
import opts
import models
# from dataloader import *
from dataloader_conceptual import *
# from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch


def eval(opt, model_name, infos_name, annFile, listener, split, iteration):
    # Input arguments and options
    # Load infos

    with open(infos_name, 'rb') as f:
        infos = cPickle.load(f, encoding='latin1')

    # For the case that we run eval not immediately after train, so arguments
    # are not exist. 'att_hid_size' is just one possible test to find out.
    if not hasattr(opt, 'att_hid_size'):
        opt = infos['opt']
    opt.split = split
    opt.beam_size = 2

    np.random.seed(123)

    # override and collect parameters
    if len(opt.input_fc_dir) == 0:
        opt.input_fc_dir = infos['opt'].input_fc_dir
        opt.input_att_dir = infos['opt'].input_att_dir
        opt.input_label_h5 = infos['opt'].input_label_h5
    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = infos['opt'].batch_size
    if len(opt.id) == 0:
        opt.id = infos['opt'].id
    # if opt.initialize_retrieval == None:
    #     opt.initialize_retrieval = infos['opt'].initialize_retrieval
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval",
              "initialize_retrieval", 'decoding_constraint',
              'evaluation_retrieval',
              "input_fc_dir", "input_att_dir", "input_label_h5", 'seq_per_img',
              'closest_num', 'closest_file']
    # for k in vars(infos['opt']).keys():
    #     if k not in ignore:
    #         if k in vars(opt) and getattr(opt, k) is not None:
    #             assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent:' + str(vars(opt)[k])+' '+ str(vars(infos['opt'])[k])
    #         else:
    #             vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping

    # assert opt.closest_num == opt.seq_per_img
    opt.vse_loss_weight = vars(opt).get('vse_loss_weight', 1)
    opt.caption_loss_weight = vars(opt).get('caption_loss_weight', 1)

    opt.cider_optimization = 0

    # Setup the model
    model = models.AlternatingJointModel(opt, iteration)
    # model = models.JointModel(opt)
    utils.load_state_dict(model, torch.load(model_name))
    if listener == 'gt':
        print('gt listener is loaded for evaluation')
        # utils.load_state_dict(model.vse, torch.load(opt.initialize_retrieval))
        utils.load_state_dict(model, {k: v for k, v in torch.load(
            opt.initialize_retrieval).items() if 'vse.' in k})

    model.cuda()
    model.eval()

    # Create the Data Loader instance
    # loader = DataLoader(opt)
    loader = DataLoader_conceptual(opt)
    # Set sample options
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, loader,
                                                                vars(opt),
                                                                annFile,
                                                                useGenSent=True)

    return {'loss': loss, 'split_predictions': split_predictions,
            'lang_stats': lang_stats}


if __name__ == '__main__':
    eval()
