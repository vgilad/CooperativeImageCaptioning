# Filename: cider.py
#
#
# Description: Describes the class to compute the CIDEr
# (Consensus-Based Image Description Evaluation) Metric
#          by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and
# Tsung-Yi Lin <tl483@cornell.edu>

from cider.pyciderevalcap.cider_diff.cider_scorer import CiderScorer
import torch


class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=4, df="corpus", word_index=None):
        """
        Initialize the CIDEr scoring function
        : param n (int): n-gram size
        : param df (string): specifies where to get the IDF values from
                    takes values 'corpus', 'coco-train'
        : return: None
        """
        # set cider to sum over 1 to 4-grams
        self._n = n
        self._df = df
        self.word_index = word_index

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        : param  gts (dict) : {image:tokenized reference sentence}
        : param res (dict)  : {image:tokenized candidate sentence}
        : return: cider (float) : computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n, word_index=self.word_index)
        for ref in gts:
            cider_scorer += ref

        # compute df on current gts
        cider_scorer.compute_df(df_mode=self._df)

        scores = []
        for i in range(res.size(0)):
            hypo = res[i]
            score = cider_scorer.compute_score(hypo, i)
            scores.append(score)

        scores = torch.stack(scores)
        return scores

    def method(self):
        return "Diff_CIDEr"
