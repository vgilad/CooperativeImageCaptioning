from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import skimage
import skimage.io
from scipy.misc import imresize
import skimage.transform
import numpy as np
import json

def if_use_att(opt):
    # Decide if load attention feature according to caption model
    if opt.caption_model in ['show_tell', 'all_img', 'fc'] and opt.vse_model \
            in ['fc', 'fc2']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.cpu().numpy())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def var_wrapper(x, cuda=True, volatile=False):
    if type(x) is dict:
        return {k: var_wrapper(v, cuda, volatile) for k,v in x.items()}
    if type(x) is list or type(x) is tuple:
        return [var_wrapper(v, cuda, volatile) for v in x]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    else:
        x = x.cpu()
    if torch.is_tensor(x):
        x = Variable(x, volatile=volatile)
    if isinstance(x, Variable) and volatile!=x.volatile:
        x = Variable(x.data, volatile=volatile)
    return x

def load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    keys = set(list(model_state_dict.keys()) + list(state_dict.keys()))
    for k in keys:
        if k not in state_dict:
            print(f'key {k} in model.state_dict() not in loaded state_dict')
        elif k not in model_state_dict:
            print(f'key {k} in loaded state_dict not in model.state_dict()')
        else:
            if state_dict[k].size() != model_state_dict[k].size():
                print(f'key {k} size not match in model.state_dict() and '
                      f'loaded state_dict. Try to flatten and copy the values '
                      f'in common parts')
            model_state_dict[k].view(-1)[:min(model_state_dict[k].numel(),
                                              state_dict[k].numel())]\
                .copy_(state_dict[k].view(-1)[:min(model_state_dict[k].numel(),
                                                   state_dict[k].numel())])

    model.load_state_dict(model_state_dict)
