from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    if torch.cuda.is_available():
        U = torch.rand(shape).cuda()
    else:  # CPU()
        U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_soft(logits, temperature, ss_prob=0.25):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    _, ind = y.max(dim=-1)
    shape = y.size()
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # one_hot = (y_hard - y).detach() + y

    if ss_prob > 0.0:  # otherwise no need to sample
        sample_prob = torch.zeros(logits.shape[0]).uniform_(0, 1)
        sample_mask = sample_prob < ss_prob

        part_logits = torch.zeros_like(logits)
        part_logits[sample_mask, :] = y[sample_mask, :]
        part_y_hard = torch.zeros_like(y_hard)
        part_y_hard[sample_mask, :] = y_hard[sample_mask, :]
        output = (part_y_hard - part_logits).detach() + y
    else:
        output = y

    return output, ind

