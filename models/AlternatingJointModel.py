#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:19:15 2018

@author: galo
"""

''' 
Explanations:

   1. {'sample_max':0} means sampling acording to the prob
   distribution normilized by temperature.

   2. {sample_max':1} means taking the argmax of the prob.
   
   3. data['gts'] is a list with len equals to batch_size, 
   each element is all captions for specific image.
   
   4. The return value from rewards.get_self_critical_reward
   (data, gen_result, greedy_res)
    is scores, cider_greedy.
    - cider_greedy - the average cider reward for the greedy results
    - scores - the difference between  cider ewards 
    for the gen_rults vs. the greedy_res
    
    5. however if return_gen_scores = True (default is False)
    then rewards.get_self_critical_reward
    (data, gen_result, greedy_res, return_gen_scores = True) 
    returns:
    cider_gen, scores, cider_greedy
    where cider_gen - vector of cider rewrds to the gen caption
    controled by the flag --use_gen_cider_scores 1
    
    6. Need a further examination - but it seems the the greedy_res are 
    genetrated based on the image and not based on random seed as 
    the article claims!!!
    
'''

import torch
import torch.nn as nn
from models import load, setup
from torch.autograd import Variable
import misc.utils as utils
import torch.nn.functional as F
import misc.rewards as rewards
import numpy as np
import copy

import os
import json
from cider.pyciderevalcap.cider_diff.cider import Cider
import misc.utils as utils


class UpdateError(Exception):
    '''
    Exception raised for error in update - meaning an update for a layer 
    that wasn't supposed to or vise vresa.

    Attributes:
        message -- explanation of the error including the name of 
        the layer what went worng in the update status
    '''

    def __init__(self, message):
        self.message = message


class AlternatingJointModel(nn.Module):
    def __init__(self, opt, iteration=None):
        super(AlternatingJointModel, self).__init__()
        self.opt = opt
        self.use_word_weights = getattr(opt, 'use_word_weights', 0)

        # self.caption_generator = setup(opt, opt.caption_model, True)
        self.caption_generator = setup(opt, opt.caption_model, 'caption_model')

        if opt.vse_model != 'None':
            # self.vse = setup(opt, opt.vse_model, False)
            self.vse = setup(opt, opt.vse_model, 'vse_model')
            self.share_embed = opt.share_embed
            if self.share_embed:
                self.caption_generator.embed[0] = self.vse.txt_enc.embed
                if self.opt.phase == 2:  # second phase (MLE) only
                    for p in self.caption_generator.embed.parameters():
                        p.requires_grad = False
        else:
            if torch.cuda.is_available():
                self.vse = lambda x, y, z, w, u: Variable(torch.zeros(1)).cuda()
            else:  # CPU()
                self.vse = lambda x, y, z, w, u: Variable(torch.zeros(1))

        if opt.retrieval_reward == 'reinforce':
            if opt.vse_loss_weight == 0 and isinstance(self.vse, nn.Module):
                for p in self.vse.parameters():
                    p.requires_grad = False


        self.batch_size = opt.batch_size
        self.vse_loss_weight = opt.vse_loss_weight
        self.caption_loss_weight = opt.caption_loss_weight
        self.df = getattr(opt, 'df', 'coco-val')
        # none, reinforce, gumbel, multinomial
        self.retrieval_reward = opt.retrieval_reward
        # In case of training listener after training speaker with
        # reinforce_speaker. optimization named reinforce_listener
        # in run_joint.sh
        if not opt.alternating_turn == None:
            if len(opt.alternating_turn) == 1 and \
                    opt.retrieval_reward == 'reinforce':
                if opt.alternating_turn[0] == 'listener':
                    opt.retrieval_reward_weight = 0
        self.retrieval_reward_weight = opt.retrieval_reward_weight  #

        self.reinforce_baseline_type = getattr(opt, 'reinforce_baseline_type',
                                               'greedy')
        self.sheriff_baseline_type = getattr(opt, 'sheriff_baseline_type',
                                             'greedy')

        self.only_one_retrieval = getattr(opt, 'only_one_retrieval', 'off')

        self.cider_optimization = getattr(opt, 'cider_optimization', 0)

        self.use_gen_cider_scores = getattr(opt, 'use_gen_cider_scores', 0)

        self._loss = {}

        # Load model
        if opt.is_alternating:  # In case of alternating training
            if opt.continue_from_existing_models:
                # If we already have a previous version of the
                # alternating model.
                if os.path.isfile(
                        os.path.join(opt.start_from, 'alternatingModel.pth')):
                    if iteration:  # For evaluation choose specific iteration
                        old_alternating_model_path = os.path.join(
                            opt.start_from, 'alternatingModel-' + iteration +
                                            '.pth')
                    else:
                        old_alternating_model_path = os.path.join(
                            opt.start_from, 'alternatingModel.pth')
                    if torch.cuda.is_available():
                        utils.load_state_dict(self, torch.load(
                            old_alternating_model_path))
                    else:
                        utils.load_state_dict(self,
                                              torch.load(
                                                  old_alternating_model_path,
                                                  map_location='cpu'))
                    print('Loaded alternating model from {}'.format(
                        old_alternating_model_path))
                else:  # initialize from stage 2 model
                    # load pre- trained speaker model from stage 2
                    old_speaker_path = opt.speaker_stage_2_model_path
                    if torch.cuda.is_available():
                        utils.load_state_dict(self,
                                              torch.load(old_speaker_path))
                    else:  # CPU()
                        utils.load_state_dict(self,
                                              torch.load(old_speaker_path,
                                                         map_location='cpu'))
                    print(f'Loaded pre-trained "speaker" model, '
                          f'after stage 2 from {old_speaker_path}')
        else:  # No alternating

            load(self, opt, iteration)
            if getattr(opt, 'initialize_retrieval', None) is not None:
                print("Make sure the vse opt are the same !!!!!")
                if torch.cuda.is_available():
                    utils.load_state_dict(self, {k: v for k, v in torch.load(
                        opt.initialize_retrieval).items() if 'vse.' in k})
                else:  # CPU
                    utils.load_state_dict(self, {k: v for k, v in torch.load(
                        opt.initialize_retrieval, map_location='cpu').items()
                                                 if 'vse.' in k})


    def getLossFlags(self):
        '''
        get the 4 flags for controlling the combination of 6 losses
        '''
        return [self.vse_loss_weight, self.caption_loss_weight,
                self.cider_optimization, self.retrieval_reward_weight]

    def setLossFlages(self, VSEWeight, MLEWeight, ciderFlag, DISCWeight):
        '''
        set the 4 flags for controlling the combination of 6 losses
        '''
        self.vse_loss_weight = VSEWeight
        self.caption_loss_weight = MLEWeight
        self.cider_optimization = ciderFlag  # Actually cider weight
        self.retrieval_reward_weight = DISCWeight

    def ce_loss(self, fc_feats, att_feats, att_masks, seq, masks):
        if self.caption_loss_weight > 0:  # MLE loss
            loss_cap = self.caption_generator(fc_feats, att_feats,
                                              att_masks, seq, masks)
            self._loss['loss_cap'] = loss_cap.data[0]
        else:
            if torch.cuda.is_available():
                loss_cap = Variable(torch.cuda.FloatTensor([0]))
            else:  # CPU()
                loss_cap = Variable(torch.FloatTensor([0]))

        return loss_cap

    def vse_loss(self, fc_feats, att_feats, seq, masks, only_one_retrieval):
        if self.vse_loss_weight > 0:  # VSE loss
            loss_vse = self.vse(fc_feats, att_feats, seq, masks,
                                only_one_retrieval=self.only_one_retrieval)
            if torch.cuda.is_available():
                self._loss['loss_vse'] = loss_vse.data[0]
            else:  # CPU()
                self._loss['loss_vse'] = loss_vse.data[0]

        else:
            if torch.cuda.is_available():
                loss_vse = Variable(torch.cuda.FloatTensor([0]))
            else:  # CPU()
                loss_vse = Variable(torch.FloatTensor([0]))

        return loss_vse

    def reinforce_disc(self, fc_feats, att_feats, att_masks):

        _seqs, _sampleLogProbs = self.caption_generator.sample(
            fc_feats, att_feats, att_masks, {'sample_max': 0,
                                             'temperature': 1})
        gen_result, sample_logprobs = _seqs, _sampleLogProbs
        _masks = torch.cat([Variable(_seqs.data.new(
            _seqs.size(0), 2).fill_(1).float()),
                            (_seqs > 0).float()[:, :-1]], 1)

        gen_masks = _masks

        _seqs = torch.cat([Variable(
            _seqs.data.new(_seqs.size(0), 1).fill_(
                self.caption_generator.vocab_size + 1)), _seqs], 1)

        retrieval_loss = self.vse(fc_feats, att_feats, _seqs,
                                  _masks, True, only_one_retrieval=
                                  self.only_one_retrieval)

        return retrieval_loss, _seqs, _masks, gen_result, \
               sample_logprobs, gen_masks


    def greedy_baseline(self, fc_feats, att_feats, att_masks,
                        retrieval_loss, _seqs,
                        _sampleLogProbs, _masks):
        if att_masks is not None:
            wrapper = [fc_feats, att_feats, att_masks]
            _seqs_greedy, _sampleLogProbs_greedy = \
                self.caption_generator.sample(
                *utils.var_wrapper(wrapper, cuda=torch.cuda.is_available(),
                                   volatile=True), opt={
                    'sample_max': 1, 'temperature': 1})
        else:
            wrapper = [fc_feats, att_feats]
            _seqs_greedy, _sampleLogProbs_greedy = \
                self.caption_generator.sample(
                    *utils.var_wrapper(wrapper, cuda=torch.cuda.is_available(),
                                       volatile=True), None, opt={
                        'sample_max': 1, 'temperature': 1})
        greedy_res = _seqs_greedy

        if (_seqs_greedy > 0).float()[:, :-1].dim() > 1:
            _masks_greedy = torch.cat([Variable(
                _seqs_greedy.data.new(_seqs.size(
                    0), 2).fill_(1).float()),
                (_seqs_greedy > 0).float()[:, :-1]], 1)
        else:
            _masks_greedy = torch.cat([Variable(
                _seqs_greedy.data.new(
                    _seqs.size(0), 2).fill_(1).float()),
                torch.unsqueeze(
                    (_seqs_greedy > 0).float()[:, :-1], 1)], 1)

        _seqs_greedy = torch.cat([Variable(
            _seqs_greedy.data.new(
                _seqs_greedy.size(0), 1).fill_(
                self.caption_generator.vocab_size + 1)),
            _seqs_greedy], 1)

        baseline = self.vse(fc_feats, att_feats,
                            _seqs_greedy, _masks_greedy,
                            True, only_one_retrieval=
                            self.only_one_retrieval)

        sc_loss = _sampleLogProbs * (utils.var_wrapper(
            retrieval_loss, cuda=torch.cuda.is_available())
                                     - utils.var_wrapper(
                    baseline, cuda=torch.cuda.is_available()
                )).detach().unsqueeze(1) * (
                      _masks[:, 1:].detach().float())
        return baseline, sc_loss, greedy_res

    def gt_baseline(self, fc_feats, att_feats, att_masks, retrieval_loss,
                    _seqs, _sampleLogProbs, _masks, seq, masks):

        baseline = self.vse(fc_feats, att_feats, seq, masks,
                            True, only_one_retrieval=self.only_one_retrieval)
        sc_loss = _sampleLogProbs * (utils.var_wrapper(
            retrieval_loss, cuda=torch.cuda.is_available
            ()) - utils.var_wrapper(baseline, cuda=
        torch.cuda.is_available())).detach() \
            .unsqueeze(1) * (_masks[:, 1:].detach().float())
        return baseline, sc_loss

    def no_baseline(self, retrieval_loss, _sampleLogProbs, _masks):

        baseline = 0
        sc_loss = _sampleLogProbs * (utils.var_wrapper(
            retrieval_loss, torch.cuda.is_available())) \
            .detach().unsqueeze(1) * (_masks[:, 1:].
                                      detach().float())
        return baseline, sc_loss

    def loss_configuration(self, loss, sc_loss, baseline, retrieval_loss,
                           _masks):

        sc_loss = sc_loss.sum() / _masks[:, 1:].data.float().sum()
        loss += self.retrieval_reward_weight * sc_loss

        self._loss['retrieval_sc_loss'] = sc_loss.data[0]
        self._loss['retrieval_loss'] = \
            retrieval_loss.sum().data[0]
        self._loss['retrieval_loss_greedy'] = baseline.sum().data[
            0] if isinstance(baseline, Variable) else baseline
        return loss

    def reinforce(self, fc_feats, att_feats, att_masks, seq, masks, data, loss):

        retrieval_loss, _seqs, _masks, gen_result, \
               sample_logprobs, gen_masks = \
            self.reinforce_disc(fc_feats, att_feats, att_masks)

        return loss, gen_result, sample_logprobs, _masks, gen_result, \
               gen_masks, _seqs, retrieval_loss,

    def st_and_ps_methods(self, fc_feats, att_feats, att_masks, seq,
                          masks, data, loss):
        # gumbel, multinomial, gumbel_softmax and multinomial_soft
        word_index, _seqs, _sampleLogProbs = self.caption_generator.sample(
            fc_feats, att_feats, att_masks,
            {'sample_max': 0, 'temperature': 1, 'use_one_hot': 1})

        # For later use for CIDEr optimization
        gen_result, sample_logprobs = word_index, _sampleLogProbs

        _masks = torch.cat([Variable(
            word_index.data.new(word_index.size(0), 2).fill_(
                1).float()), (word_index > 0).float()[:, :-1]], 1)
        gen_masks = _masks  # For later use for CIDEr optimization
        # indices to one-hot vector
        if torch.cuda.is_available():
            # add manually BOS token to _seqs
            one_hot_bos_token = \
                torch.zeros(self.opt.batch_size, 1,
                            self.caption_generator.vocab_size + 2).cuda()
        else:  # CPU()
            # add manually BOS token to _seqs
            one_hot_bos_token = torch.zeros(self.opt.batch_size, 1,
                                            self.caption_generator.
                                            vocab_size + 2)
        one_hot_bos_token[:, 0,
        self.caption_generator.vocab_size + 1] = 1
        _seqs = torch.cat([one_hot_bos_token, _seqs], 1)
        loss_vse = self.vse(fc_feats, att_feats, _seqs, _masks,
                            only_one_retrieval=
                            self.only_one_retrieval)
        loss += loss_vse * self.retrieval_reward_weight

        return loss, gen_result, sample_logprobs, gen_masks, _seqs

    def gen_result_for_cider(self, fc_feats, att_feats, att_masks):

        gen_result, sample_logprobs = \
            self.caption_generator.sample(
                fc_feats, att_feats, att_masks,
                opt={'sample_max': 0})

        gen_masks = torch.cat([Variable(
            gen_result.data.new(gen_result.size(0), 2).fill_(
                1).float()), (gen_result > 0).float()[:, :-1]], 1)

        return gen_result, sample_logprobs, gen_masks

    def greedy_res_for_cider(self, fc_feats, att_feats, att_masks):
        if att_masks is not None:
            greedy_res, _ = self.caption_generator.sample(
                *utils.var_wrapper([fc_feats, att_feats, att_masks],
                                   cuda=torch.cuda.is_available(),
                                   volatile=True),
                opt={'sample_max': 1})
        else:
            greedy_res, _ = self.caption_generator.sample(*utils.var_wrapper(
                [fc_feats, att_feats], cuda=torch.cuda.is_available(),
                volatile=True), att_masks, opt={'sample_max': 1})

        return greedy_res

    def traditional_cider(self, fc_feats, att_feats, att_masks, data, loss,
                          gen_result, greedy_res, sample_logprobs,
                          gen_masks):

        # Use the differenced rewards
        if self.use_gen_cider_scores == 0:
            reward, cider_greedy = rewards.get_self_critical_reward(
                data, gen_result, greedy_res)
        else:  # use the original rewards
            reward, _, cider_greedy = \
                rewards.get_self_critical_reward(
                    data, gen_result, greedy_res,
                    return_gen_scores=True)
        self._loss['avg_reward'] = reward.mean()
        self._loss['cider_greedy'] = cider_greedy

        loss_cider = sample_logprobs * utils.var_wrapper(
            -reward.astype('float32'),
            cuda=torch.cuda.is_available()).unsqueeze(1) * (
                         gen_masks[:, 1:].detach())

        loss_cider = loss_cider.sum() / \
                     gen_masks[:, 1:].data.float().sum()
        loss += self.cider_optimization * loss_cider
        self._loss['loss_cider'] = loss_cider.data[0]

        return loss

    def forward(self, fc_feats, seq, masks, data, att_feats, att_masks, \
                is_alternating=False, alternating_turn=None):
        '''
        seq - ground truth captions
        is_alternating - whether to perform alternating training, 
        speaker & listener are trained in turns [True / False] 
        alternating_turn - whose turn is it to be trained 
        (relevant only in case is_alternating = true) ['speaker' / 'listener']
        '''

        if not is_alternating:  # regular unAlternating training

            # (used for training only one model)
            # Not composing mutual exclusivness between cider & MLE loss
            loss_cap = self.ce_loss(fc_feats, att_feats, att_masks, seq, masks)
            loss_vse = self.vse_loss(fc_feats, att_feats, seq, masks,
                                    only_one_retrieval=self.only_one_retrieval)

            loss = self.caption_loss_weight * loss_cap + \
                   self.vse_loss_weight * loss_vse

            # DISC loss
            if (self.retrieval_reward_weight > 0):
                if self.retrieval_reward == 'reinforce':
                    # Calculate reward
                    loss, gen_result, sample_logprobs, _masks, gen_result, \
                    gen_masks, _seqs, retrieval_loss = \
                        self.reinforce(fc_feats, att_feats, att_masks, seq,
                                       masks, data, loss)
                    # Baseline
                    if self.reinforce_baseline_type == 'greedy':
                        baseline, sc_loss, greedy_res = self.greedy_baseline(
                            fc_feats, att_feats, att_masks, retrieval_loss,
                            _seqs, sample_logprobs, _masks)

                    elif self.reinforce_baseline_type == 'gt':
                        baseline, sc_loss = \
                            self.gt_baseline(fc_feats, att_feats, att_masks,
                                             retrieval_loss, _seqs,
                                             sample_logprobs, _masks, seq,
                                             masks)

                    else:  # No Baseline
                        baseline, sc_loss = \
                            self.no_baseline(retrieval_loss, sample_logprobs,
                                             _masks)

                    loss = self.loss_configuration(loss, sc_loss, baseline,
                                                   retrieval_loss, _masks)

                else:
                    # gumbel, multinomial, gumbel_softmax, multinomial_soft
                    loss, gen_result, sample_logprobs, gen_masks, _seqs = \
                        self.st_and_ps_methods(
                            fc_feats, att_feats, att_masks, seq, masks,
                                                  data, loss)
            # CIDER loss
            if self.cider_optimization:
                if 'gen_result' not in locals() or self.retrieval_reward in \
                        ['multinomial_soft', 'gumbel_softmax']:
                    gen_result, sample_logprobs, gen_masks = \
                        self.gen_result_for_cider(fc_feats, att_feats,
                                         att_masks)

                if 'greedy_res' not in locals():
                    greedy_res = self.greedy_res_for_cider(
                        fc_feats, att_feats, att_masks)

                loss = self.traditional_cider(
                    fc_feats, att_feats, att_masks, data, loss,
                    gen_result, greedy_res, sample_logprobs, gen_masks)
            return loss


        else:  # Alternating training
            if alternating_turn == 'speaker':  # speaker's turn
                # Changing to suitable update status
                gradientsDic = {'vseModel': False, 'captionModel': True}
                if self.retrieval_reward == 'reinforce':
                    self.changeModelUpdateStatus(gradientsDic,
                                                 printWeights=False)
                # Calling to the non- alternating version with suitible
                # parameters
                oldVSE, oldMLE, oldCider, oldDISC = self.getLossFlags()
                self.setLossFlages(VSEWeight=0, MLEWeight=oldMLE,
                                   ciderFlag=oldCider, DISCWeight=oldDISC)
                newVSE, newMLE, newCider, newDISC = self.getLossFlags()
                loss = self.forward(fc_feats, seq, masks, data, att_feats,
                                    att_masks, is_alternating=False,
                                    alternating_turn=alternating_turn)
                # Change back the parameters
                self.setLossFlages(VSEWeight=oldVSE, MLEWeight=oldMLE,
                                   ciderFlag=oldCider, DISCWeight=oldDISC)
                return loss

            elif alternating_turn == 'listener':  # listener's turn
                # Changing to suitable update status
                gradientsDic = {'vseModel': True, 'captionModel': False}
                self.changeModelUpdateStatus(gradientsDic, printWeights=False)
                # Calling to the non- alternating version with suitable
                # parameters
                oldVSE, oldMLE, oldCider, oldDISC = self.getLossFlags()
                self.setLossFlages(VSEWeight=oldVSE, MLEWeight=0, ciderFlag=0,
                                   DISCWeight=0)
                newVSE, newMLE, newCider, newDISC = self.getLossFlags()
                # Generating captions for training listener
                _seqs, _sampleLogProbs = self.caption_generator.sample(
                    fc_feats, att_feats, att_masks,
                    {'sample_max': 0, 'temperature': 1})
                _masks = torch.cat([Variable(
                    _seqs.data.new(_seqs.size(0), 2).fill_(1).float()),
                    (_seqs > 0).float()[:, :-1]], 1)
                _seqs = torch.cat([Variable(_seqs.data.new(
                    _seqs.size(0), 1).fill_(self.caption_generator.vocab_size +
                                            1)), _seqs], 1)
                # Calling non-alternating version with generated captions
                loss = self.forward(fc_feats, _seqs, _masks, data, att_feats,
                                    att_masks, is_alternating=False,
                                    alternating_turn=alternating_turn)
                # Change back the parameters
                self.setLossFlages(VSEWeight=oldVSE, MLEWeight=oldMLE,
                                   ciderFlag=oldCider, DISCWeight=oldDISC)
                return loss

    def sample(self, fc_feats, att_feats, att_masks, opt={}):

        return self.caption_generator.sample(fc_feats, att_feats, att_masks,
                                             opt)

    def loss(self):
        out = {}
        out.update(self._loss)
        out.update(
            {'cap_' + k: v for k, v in self.caption_generator._loss.items()})
        out.update({'vse_' + k: v for k, v in self.vse._loss.items()})
        return out


    def changeModelUpdateStatus(self, gradDic, printWeights=False):

        '''
        gradDic - a dictionary with submodels names as keys & boolean values
        for the respective required_grad
        printWeights - whether of not to print the weights of subModel
        '''
        print(f'required grad status is {gradDic}')
        # Checks whether the previous parameters are saved,
        # save them if they aren't
        if (not (hasattr(self, 'prev_gradDic')) or not (
                hasattr(self, 'prev_vse')) \
                or not (hasattr(self, 'prev_caption_generator'))):
            self.prev_gradDic = copy.deepcopy(gradDic)
            self.prev_vse = copy.deepcopy(self.vse)
            self.prev_caption_generator = copy.deepcopy(self.caption_generator)
            doCompareToPrev = False
        else:
            doCompareToPrev = True

        # VSE part
        if ('vseModel' in gradDic.keys()):
            layerNum = 0
            equalToPrev = None
            for (curr_name, curr_p), (prev_name, prev_p) in zip(
                    self.vse.named_parameters(),
                    self.prev_vse.named_parameters()):
                curr_p.requires_grad = gradDic['vseModel']
                if printWeights:
                    print(f'vseModel {layerNum + 1} layer name is {curr_name}')
                    print(f'vseModel {layerNum + 1} layer data is '
                          f'{curr_p.data}')
                    print(f'previos vseModel {layerNum + 1} '
                          f'layer name is {prev_name}')
                    print(f'previos vseModel {layerNum + 1} '
                          f'layer data is {prev_p.data}')
                layerNum += 1
                # Checking for consistence among the updates of the different
                # model's layers
                if (doCompareToPrev):
                    if (equalToPrev is None):
                        equalToPrev = (float(
                            (curr_p.data - prev_p.data).norm()) == 0)
                    elif (equalToPrev != (
                            float((curr_p.data - prev_p.data).norm()) == 0)):
                        try:
                            raise UpdateError(
                                f'WARNNING - layer {curr_name} in vse model '
                                f'had inconsistent update status with the '
                                f'rest of the model layers')
                        except UpdateError as err:
                            print(err.message)
            # checking whether the model was updated according to previous
            # status
            if doCompareToPrev:
                if equalToPrev == self.prev_gradDic['vseModel']:
                    try:
                        raise UpdateError('WARNNING - vseModel update status '
                                          'was supposed to be {} but in fact '
                                          'was {}' .format(
                            self.prev_gradDic['vseModel'], not (equalToPrev)))
                    except UpdateError as err:
                        print(err.message)
        # CAPTION part
        if 'captionModel' in gradDic.keys():
            layerNum = 0
            equalToPrev = None
            for (curr_name, curr_p), (prev_name, prev_p) in zip(
                    self.caption_generator.named_parameters(),
                    self.prev_caption_generator.named_parameters()):
                curr_p.requires_grad = gradDic['captionModel']
                if printWeights:
                    print(f'captionModel {layerNum + 1} layer name is '
                          f'{curr_name}')
                    print(f'captionModel {layerNum + 1} layer data is '
                          f'{curr_p.data}')
                    print(f'previos captionModel {layerNum + 1} '
                          f'layer name is {prev_name}')
                    print(f'previos captionModel {layerNum + 1} layer data '
                          f'is {prev_p.data}')
                layerNum += 1
                # Checking for consistence among the updates of the different
                # model's layers
                if doCompareToPrev:
                    if equalToPrev is None:
                        equalToPrev = (
                                float((curr_p.data - prev_p.data).norm()) == 0)
                    elif (equalToPrev != (
                            float((curr_p.data - prev_p.data).norm()) == 0)):
                        try:
                            raise UpdateError('WARNNING - layer {} in caption '
                                              'model had inconsistent update '
                                              'status with the rest of the '
                                              'model layers'
                                              .format(curr_name))
                        except UpdateError as err:
                            print(err.message)
            # Checking whether the model was updated according to previous
            # status
            if doCompareToPrev:
                if equalToPrev == self.prev_gradDic['captionModel']:
                    try:
                        raise UpdateError('WARNNING - captionModel update '
                                          'status was supposed to be {} but in '
                                          'fact was {}'.format(
                            self.prev_gradDic['captionModel'], not equalToPrev))
                    except UpdateError as err:
                        print(err.message)

            # saving the current values as prev for next iteration
            if doCompareToPrev:
                self.prev_gradDic = copy.deepcopy(gradDic)
                self.prev_vse = copy.deepcopy(self.vse)
                self.prev_caption_generator = copy.deepcopy(self.
                                                            caption_generator)

