from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.norm(X, dim=1, keepdim=True) + 1e-7
    X = torch.div(X, norm)
    return X

class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.vse_embed_size
        self.no_imgnorm = opt.vse_no_imgnorm
        self.use_abs = opt.vse_use_abs
        self.fc_feat_size = opt.fc_feat_size

        self.fc = nn.Linear(self.fc_feat_size, self.embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # Normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # Take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.use_abs = opt.vse_use_abs
        self.input_encoding_size = opt.input_encoding_size
        self.embed_size = opt.vse_embed_size
        self.num_layers = opt.vse_num_layers
        self.rnn_type = opt.vse_rnn_type
        self.vocab_size = opt.vocab_size
        self.use_abs = opt.vse_use_abs
        self.pool_type = getattr(opt, 'vse_pool_type', '')
        # Word embedding
        self.embed = nn.Embedding(self.vocab_size + 2,
                                  self.input_encoding_size)

        # Caption embedding
        self.rnn = getattr(nn, self.rnn_type.upper())(
            self.input_encoding_size, self.embed_size, self.num_layers,
            batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def pad_sentences(self, seqs, masks):
        len_sents = (masks > 0).long().sum(1)
        len_sents, len_ix = len_sents.sort(0, descending=True)

        inv_ix = len_ix.clone()
        inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(
            inv_ix.data)

        new_seqs = seqs[len_ix].contiguous()

        return new_seqs, len_sents, len_ix, inv_ix

    def forward(self, seqs, masks):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        padded_seqs, sorted_lens, len_ix, inv_ix = self.pad_sentences(
            seqs, masks)

        if seqs.dim() > 2:
            # One hot input
            seqs_embed = torch.matmul(padded_seqs, self.embed.weight)
        else:
            seqs_embed = self.embed(padded_seqs)

        seqs_pack = pack_padded_sequence(seqs_embed, list(sorted_lens.data),
                                         batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(seqs_pack)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = sorted_lens.view(-1, 1, 1).expand(
            seqs.size(0), 1, self.embed_size) - 1
        if self.pool_type == 'mean':
            out = padded[0]
            _masks = masks[len_ix].float()
            out = (out * _masks[:, :out.size(1)].unsqueeze(-1)).sum(
                1) / _masks.sum(1, keepdim=True)
        elif self.pool_type == 'max':
            out = padded[0]
            _masks = masks[len_ix][:,:out.size(1)].float()
            out = (out * _masks.unsqueeze(-1) + (_masks == 0).unsqueeze(
                -1).float() * -1e10).max(1)[0]
        else:
            out = padded[0].gather(1, I).squeeze(1)

        # Normalization in the joint embedding space
        out = l2norm(out)

        out = out[inv_ix].contiguous()

        # Take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.margin = opt.vse_margin
        self.measure = opt.vse_measure
        if self.measure == 'cosine':
            self.sim = cosine_sim
        else:
            print("Warning: Similarity measure not supported: {}".format(
                self.measure))
            self.sim = None

        self.max_violation = opt.vse_max_violation

    def forward(self, im, s, whole_batch=False, only_one_retrieval='off'):
        # Compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # Compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # Compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # Clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # Keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        else:
            cost_s = cost_s.mean(1)
            cost_im = cost_im.mean(0)

        if whole_batch:
            fn = lambda x: x
        else:
            fn = lambda x: x.sum()

        if only_one_retrieval == 'image':
            return fn(cost_im)
        elif only_one_retrieval == 'caption':
            return fn(cost_s)
        else:
            return fn(cost_s) + fn(cost_im)


class VSEFCModel(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        super(VSEFCModel, self).__init__()
        # Build Models
        self.loss_type = opt.vse_loss_type

        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)
        # Loss and Optimizer
        self.contrastive_loss = ContrastiveLoss(opt)

        self.margin = opt.vse_margin
        self.embed_size = opt.vse_embed_size

        self._loss = {}

    def forward(self, fc_feats, att_feats, seq, masks, whole_batch=False,
                only_one_retrieval='off'):

        img_emb = self.img_enc(fc_feats)
        cap_emb = self.txt_enc(seq, masks)

        loss = self.contrastive_loss(img_emb, cap_emb, whole_batch,
                                     only_one_retrieval)
        if not whole_batch:
            self._loss['contrastive'] = loss.data[0]

        return loss

