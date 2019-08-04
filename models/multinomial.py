from __future__ import print_function
import torch

def multinomial(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    if temperature == 1:
        y = torch.softmax(logits, 1)
    else:
        # y is the probability that we want to backprop through
        # BUGFIXED, but not tested
        y = torch.softmax(torch.div(logits, temperature), 1)
    # Choose index by raffle on the probability y
    ind = torch.multinomial(y, 1).squeeze(1)
    shape = y.size()
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape) # y_hard is one-hot of word index

    # this trick allows to pass one-hot on forward, because y_hard =
    # y_hard - y + y and softmax (y) on backward, because detach stop the
    # gradients from flowing to (y_hard - y)
    one_hot = (y_hard - y).detach() + y

    return one_hot, ind

