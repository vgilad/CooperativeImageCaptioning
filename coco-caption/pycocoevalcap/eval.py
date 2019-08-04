__author__ = 'tylin'
import sys
sys.path.append('coco-caption')
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap import eval_spice
import os
import json

class COCOEvalCap:
    def __init__(self, coco, cocoRes, ckpt_path):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}
        self.ckpt_path = ckpt_path

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
           (Spice(), "SPICE")
         ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            if method == "SPICE":

                gts_name = self.ckpt_path + '/gts.json'
                res_name = self.ckpt_path + '/res.json'
                with open(gts_name, 'w') as f:
                    json.dump(gts, f)
                with open(res_name, 'w') as f:
                    json.dump(res, f)
                cmd = "/cortex/users/gilad/anaconda3/envs/discaption27/bin" \
                      "/python2.7 " \
                      "/home/lab/vgilad/PycharmProjects/JointImageCaptioning" \
                      "/coco-caption/pycocoevalcap/eval_spice.py " + \
                      "--ckpt_path " + self.ckpt_path
                os.system(cmd)
                with open(self.ckpt_path + '/score.json') as f:
                    score = json.load(f)
                with open(self.ckpt_path + '/scores.json') as f:
                    scores = json.load(f)
                os.remove(gts_name)
                os.remove(res_name)
                os.remove(self.ckpt_path + '/score.json')
                os.remove(self.ckpt_path + '/scores.json')
            else:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for imgId in sorted(self.imgToEval.keys())]
