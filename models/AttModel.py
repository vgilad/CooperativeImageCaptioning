# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, \
    pad_packed_sequence
from models.gumbel import gumbel_softmax
from models.gumbel_softmax import gumbel_soft
from models.multinomial import multinomial
from models.multinomial_soft import multinomial_soft

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.tolist(),
                               batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(
            att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]),
                                                         packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(nn.Module):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.retrieval_reward = opt.retrieval_reward
        self.gumbel_temp = opt.gumbel_temp
        self.multinomial_temp = opt.multinomial_temp
        self.prob_gumbel_softmax = getattr(opt, 'prob_gumbel_softmax', 1)
        self.prob_multinomial_soft = getattr(opt, 'prob_multinomial_soft', 1)
        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(
            nn.Embedding(self.vocab_size + 2, self.input_encoding_size),
            nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.relu_dropout = nn.Sequential(nn.ReLU(),
                                          nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(
            nn.Linear(self.fc_feat_size, self.rnn_size), nn.ReLU(), nn.Dropout(
                self.drop_prob_lm))
        self.att_embed = nn.Sequential(
            *(((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
              (nn.Linear(self.att_feat_size, self.rnn_size), nn.ReLU(),
               nn.Dropout(self.drop_prob_lm))))

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        
        self.crit = utils.LanguageModelCriterion()

        self.decoding_constraint = getattr(opt, 'decoding_constraint', 0)

        self._loss = {}

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(
            weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(
                    self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, fc_feats, att_feats, att_masks, seq, masks):

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        # Embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce
        # memory and computation consumptions.
        p_att_feats = self.ctx2att(att_feats)

        for i in range(seq.size(1) - 1):
            # If condition below is False no need to sample
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)
                    it.index_copy_(0, sample_ind, torch.multinomial(
                        prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(
                xt, fc_feats, att_feats, p_att_feats, att_masks, state)
            output = F.log_softmax(self.logit(output))
            outputs.append(output)

        output = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()
        loss = self.crit(output, seq[:,1:], masks[:,1:])

        self._loss['xe'] = loss.data[0]

        return loss

    def sample_beam(self, fc_feats, att_feats, att_masks, opt={}):

        beam_size = opt.get('beam_size', 10)
        decoding_constraint = opt.get('decoding_constraint',
                                      self.decoding_constraint)
        batch_size = fc_feats.size(0)

        # Embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and
        # computation consumptions.
        p_att_feats = self.ctx2att(att_feats)
        
        assert beam_size <= self.vocab_size + 1, \
            'lets assume this for now, otherwise this corner case causes a ' \
            'few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(
                *((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(
                *((beam_size,)+p_att_feats.size()[1:])).contiguous()
            if att_masks is not None:
                tmp_att_masks = att_masks[k:k+1].expand(
                    *((beam_size,)+att_masks.size()[1:])).contiguous()
            else:
                tmp_att_masks = None
            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(
                self.seq_length, beam_size).zero_()
            # Running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)
            done_beams = []
            for t in range(self.seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().fill_(
                        self.vocab_size + 1)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """pem a beam merge. that is,
                    for every previous beam we now many new possibilities 
                    to branch out we need to resort our beams to maintain the 
                    loop invariant of keeping the top beam_size most likely 
                    sequences."""
                    # Lets go to CPU for more efficiency in indexing operations
                    logprobsf = logprobs.float().cpu()
                    if decoding_constraint and t > 1:
                        tmp = logprobsf.data.new(logprobsf.size()).zero_()
                        tmp.scatter_(1, beam_seq[t-2:t-1].t(), float('-inf'))
                        logprobsf = logprobsf + Variable(tmp)
                    # Sorted array of logprobs along each previous beam (
                    # last true = descending)
                    ys,ix = torch.sort(logprobsf,1,True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    # At first time step only the first beam is active
                    if t == 1:
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # Compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + \
                                                local_logprob
                            candidates.append({'c': ix.data[q, c], 'q': q,
                                               'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})

                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # Construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        # Well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # Fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t-1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-1, vix] = \
                                beam_seq_logprobs_prev[:, v['q']]

                        # Rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # Copy over state in previous beam q to new beam
                            # at vix
                            # dimension one is time step
                            new_state[state_ix][0, vix] = \
                                state[state_ix][0, v['q']]

                        # Append new end terminal at the end of this beam
                        # C'th word is the continuation
                        beam_seq[t-1, vix] = v['c']
                        # The raw logprob here
                        beam_seq_logprobs[t-1, vix] = v['r']
                        # The new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or t == self.seq_length:
                            # END token special case here,
                            # or we reached the end.
                            # Add the beam to a set of done beams
                            self.done_beams[k].append(
                                {'seq': beam_seq[:, vix].clone(),
                                 'logps': beam_seq_logprobs[:, vix].clone(),
                                 'p': beam_logprobs_sum[vix]})
        
                    # Encode as vectors
                    it = beam_seq[t-1]
                    if torch.cuda.is_available():
                        xt = self.embed(Variable(it.cuda()))
                    else:  # CPU()
                        xt = self.embed(Variable(it))
                
                if t >= 1:
                    state = new_state

                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats,
                                          tmp_p_att_feats, tmp_att_masks, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = \
                sorted(self.done_beams[k], key=lambda x: -x['p'])
            # The first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # Return the samples and their log likelihoods
        return Variable(
            seq.transpose(0, 1)), Variable(seqLogprobs.transpose(0, 1))

    def sample(self, fc_feats, att_feats, att_masks, opt={}):
        use_one_hot = opt.get('use_one_hot', 0)
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint',
                                      self.decoding_constraint)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # Embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory
        # and computation consumptions.
        p_att_feats = self.ctx2att(att_feats)
        word_index = []
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # Input <bos>
                it = Variable(fc_feats.data.new(batch_size).long()
                              .fill_(self.vocab_size + 1))
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs, 1)
                it = it.view(-1).long()
            # If True, generate a word index, otherwise, generate
            # a soft distribution on words
            elif self.retrieval_reward == 'reinforce' or not use_one_hot:
                if temperature == 1.0:
                    # Fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(logprobs)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs, temperature))
                it = torch.multinomial(prob_prev, 1).detach()
                # Gather the logprobs at sampled positions
                sampleLogprobs = logprobs.gather(1, it)
                # Flatten indices for downstream processing
                it = it.view(-1).long()
            # Straight Through Gumbel-Softmax
            elif self.retrieval_reward == 'gumbel':
                one_hot, it = gumbel_softmax(logprobs, self.gumbel_temp)
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
                if torch.cuda.is_available():
                    one_hot = torch.cat([one_hot, torch.zeros(one_hot.size(0),
                                                              1).cuda()], 1)
                else:
                    one_hot = torch.cat([one_hot,
                                         torch.zeros(one_hot.size(0),
                                                     1)], 1)
            # Straight Through Multinomial
            elif self.retrieval_reward == 'multinomial':
                one_hot, it = multinomial(logprobs, self.multinomial_temp)
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
                if torch.cuda.is_available():
                    one_hot = torch.cat([one_hot, torch.zeros(one_hot.size(0),
                                                              1).cuda()], 1)
                else:
                    one_hot = torch.cat([one_hot,
                                         torch.zeros(one_hot.size(0),
                                                     1)], 1)
            # Partial-sampling Gumbel-Softmax
            elif self.retrieval_reward == 'gumbel_softmax':
                soft_vec, it = gumbel_soft(logprobs, self.gumbel_temp,
                                           self.prob_gumbel_softmax)
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
                # Add last column of zeros for <BOS> and keeping
                # the FC layer the same
                if torch.cuda.is_available():
                    soft_vec = torch.cat(
                        [soft_vec, torch.zeros(soft_vec.size(0), 1).cuda()], 1)
                else:
                    soft_vec = torch.cat([soft_vec, torch.zeros(
                        soft_vec.size(0), 1)], 1)

            # Partial-sampling Multinomial Straight Through
            elif self.retrieval_reward == 'multinomial_soft':
                soft_vec, it = multinomial_soft(logprobs, self.multinomial_temp,
                                           self.prob_multinomial_soft)
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
                # Add last column of zeros for <BOS> and keeping
                # the FC layer the same
                if torch.cuda.is_available():
                    soft_vec = torch.cat(
                        [soft_vec, torch.zeros(soft_vec.size(0), 1).cuda()], 1)
                else:
                    soft_vec = torch.cat([soft_vec, torch.zeros(
                        soft_vec.size(0), 1)], 1)

            # 'gumbel_softmax' or 'multinomial_soft'
            if 'soft_vec' in locals():
                xt = torch.matmul(soft_vec, self.embed[0].weight)
                xt = self.relu_dropout(xt)
            else:
                xt = self.embed(it)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.data.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                # Reinforce or sample_max scenario (also with gumbel)
                if self.retrieval_reward == 'reinforce' or not use_one_hot:
                    seq.append(it) #seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))
                elif self.retrieval_reward in ['gumbel', 'multinomial']:
                    word_index.append(it)  # seq[t] the input of t+2 time step
                    one_hot = one_hot * unfinished.unsqueeze(1).\
                        type_as(it.type(torch.float32))
                    seq.append(one_hot)  # seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))

                elif self.retrieval_reward in ['gumbel_softmax',
                                               'multinomial_soft']:
                    word_index.append(it)  # seq[t] the input of t+2 time step
                    soft_vec = soft_vec * unfinished.unsqueeze(1).\
                        type_as(it.type(torch.float32))
                    seq.append(soft_vec)  # seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats,
                                      att_masks, state)
            if decoding_constraint and len(seq) > 0:
                tmp = output.data.new(output.size(0), self.vocab_size + 1)\
                    .zero_()
                tmp.scatter_(1, seq[-1].data.unsqueeze(1), float('-inf'))
                logprobs = F.log_softmax(self.logit(output)+Variable(tmp))
            else:
                logprobs = F.log_softmax(self.logit(output))
        if self.retrieval_reward == 'reinforce' or not use_one_hot:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), \
                   torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        elif self.retrieval_reward in ['gumbel', 'multinomial',
                                       'gumbel_softmax', 'multinomial_soft']:
            return torch.cat([_.unsqueeze(1) for _ in word_index], 1), \
                   torch.cat([_.unsqueeze(1) for _ in seq], 1), \
                   torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)



class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)  # batch * att_hid_size
        # att_h below is batch * att_size * # att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        # dot below is (batch * att_size) * att_hid_size
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        
        weight = F.softmax(dot)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # Normalize to 1
        # att_feats_ size is  batch * att_size * att_feat_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)
        # att_res size is batch * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res


class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, state):
        att_res = self.attention(state[0][-1],
                                 att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(
            1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c(att_res)
        in_transform = \
            torch.max(in_transform.narrow(1, 0, self.rnn_size),
                      in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x

