#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

# Modified by avahdat

import copy
import pickle
from collections import defaultdict
import numpy as np
import math
import os


import torch


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def compute_doc_freq(refs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''

    document_frequency = defaultdict(float)
    for refs in refs:
        # refs, k ref captions of one image
        #for ngram in set([ngram for ref in refs for (ngram, count) in ref.iteritems()]):  # python 2
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):  # python 3
            document_frequency[ngram] += 1

    return document_frequency


class CiderScorer(object):
    # we can avoid loading this in every iteration
    try:
        with open(os.path.join('cider/data', 'coco-val' + '.p'), 'rb') as f:
            coco_val_df = pickle.load(f, encoding='latin1')
    except:
        with open(os.path.join('../cider/data', 'coco-val' + '.p'), 'rb') as f:
            coco_val_df = pickle.load(f, encoding='latin1')
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # coco_val_df = u.load()

    """CIDEr scorer.
    """
    def __init__(self, word_index=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.cook_append(refs)
        self.word_index = word_index

    def cook_append(self, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''
        if refs is not None:
            self.crefs.append(cook_refs(refs))

    def size(self):
        return len(self.crefs)

    def __iadd__(self, refs):
        '''add an instance (e.g., from another sentence).'''
        self.cook_append(refs)
        return self

    def compute_df(self, df_mode=None):
        # compute idf
        if df_mode == "corpus":
            self.df = compute_doc_freq(self.crefs)
            self.ref_len = np.log(float(len(self.crefs)))

            # assert to check document frequency
            assert(len(self.crefs) >= max(self.df.values()))
        elif df_mode == 'coco-val':
            self.df = CiderScorer.coco_val_df
            self.ref_len = np.log(float(40504))
        else:
            raise NotImplementedError

    def compute_cider(self, res, index):
        def sparse_ngrams_matrix(ref):
            vocab_size = len(self.word_index) + 1

            indices = [[[] for m in range(n+1)] for n in range(self.n)]
            values = [[[] for m in range(n+1)] for n in range(self.n)]
            count = [0] * self.n

            tf_ref = [[] for n in range(self.n)]
            df_ref = [[] for n in range(self.n)]
            for (ref_ngram, ref_term_freq) in ref.items():
                n = len(ref_ngram) - 1
                for w_ind, w in enumerate(ref_ngram):
                    ind = self.word_index[w] if w in self.word_index else len(self.word_index)
                    # indices[n][w_ind].append([count[n], ind]) #  Original
                    indices[n][w_ind].append([count[n], int(ind)])

                    values[n][w_ind].append(1.)
                count[n] += 1

                # compute term frequency for reference
                tf_ref[n].append(ref_term_freq)

                # compute df
                df_ref[n].append(np.log(max(1.0, self.df[ref_ngram])))

            sparse_matrices = [[] for n in range(self.n)]
            tf_ref_np = []
            df_ref_np = []
            for n in range(self.n):
                for m in range(n+1):
                    if len(indices[n][m]) == 0:
                        break

                    i = torch.LongTensor(indices[n][m])
                    v = torch.FloatTensor(values[n][m])
                    sparse_matrices[n].append(torch.sparse.FloatTensor(i.t(), v, torch.Size([count[n], vocab_size])))

                tf_ref_np.append(np.array(tf_ref[n], dtype=np.float32))
                df_ref_np.append(np.array(df_ref[n], dtype=np.float32))

            return sparse_matrices, tf_ref_np, df_ref_np

        def diff_sim(res, ref):
            ngram_matrices, ref_frequencies, df = sparse_ngrams_matrix(ref)

            score = 0.
            count = 0
            for n in range(len(ngram_matrices)):
                if len(ngram_matrices[n]) == 0:
                    continue
                if torch.cuda.is_available():
                    res_freq = torch.zeros(ngram_matrices[n][0].size(0),
                                           res.size(1)).cuda()
                else:  # CPU()
                    res_freq = torch.zeros(ngram_matrices[n][0].size(0),
                                           res.size(1))

                for m in range(n+1):
                    # mult = torch.sparse.mm(ngram_matrices[n][m], res)
                    if torch.cuda.is_available():
                        mult = torch.mm(ngram_matrices[n][m].cuda(), res)
                    else:  # CPU()
                        mult = torch.mm(ngram_matrices[n][m], res)
                    if m == 0:
                        res_freq += mult
                    else:
                        res_freq[:, :-m] += mult[:, m:]

                # res_freq = torch.sparse.mm(ngram_matrices[n], shifted_res[n].t().clone())
                res_freq = torch.exp(res_freq / (n + 1))

                res_freq = torch.sum(res_freq, dim=1)
                res_freq /= (res.size(1) - n)

                ref_freq = torch.from_numpy(ref_frequencies[n])
                ref_freq /= torch.sum(ref_freq)

                idf = torch.from_numpy(self.ref_len - df[n])

                # histogram intersection
                if torch.cuda.is_available():
                    score += torch.sum(torch.min(res_freq, ref_freq.cuda()) *
                                       idf.cuda())
                else:  # CPU()
                    score += torch.sum(torch.min(res_freq, ref_freq) * idf)
                count += 1
            return score / count

        res = torch.log(res + 1e-20)
        transposed_res = res.t().clone()
        # shifted_res = shift_candidate(res)
        # compute vector for ref captions
        score = 0.
        for ref in self.crefs[index]:
            score += diff_sim(transposed_res, ref)

        # divide by number of references
        score /= len(self.crefs[index])
        # multiply score by 10
        score *= 10.0
        # append score of an image to the score list

        return score

    def compute_score(self, res, index):
        # compute cider score
        score = self.compute_cider(res, index)
        # debug
        # print score
        return score
