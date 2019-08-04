from __future__ import print_function
import torch


def multinomial_soft(logits, temperature, ss_prob=0.25):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """

    if temperature == 1:
        y = torch.exp(logits)
    else:
        # y is the probability that we want to backprop through
        y = torch.exp(torch.div(logits, temperature))
    # Choose index by raffle on the probability y
    ind = torch.multinomial(y, 1).squeeze(1)
    shape = y.size()
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    if ss_prob > 0.0:  # otherwise no need to sample
        sample_prob = torch.zeros(logits.shape[0]).uniform_(0, 1)
        sample_mask = sample_prob < ss_prob

        part_prob = torch.zeros_like(logits)
        part_prob[sample_mask, :] = y[sample_mask, :]
        part_y_hard = torch.zeros_like(y_hard)
        part_y_hard[sample_mask, :] = y_hard[sample_mask, :]
        output = (part_y_hard - part_prob).detach() + y
    else:
        output = y

    return output, ind

