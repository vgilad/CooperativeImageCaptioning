__author__ = 'tylin'
import sys
sys.path.append('coco-caption')

try:
    from pycocoevalcap.spice.spice import Spice
except:
    from .spice.spice import Spice
import json

def main(ckpt_path, gts_name='/gts.json', res_name='/res.json'):
    print("eval_spice.py")
    print(ckpt_path)
    with open(ckpt_path + gts_name) as f:
        gts = json.load(f)
    with open(ckpt_path + res_name) as f:
        res = json.load(f)
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    with open(ckpt_path + '/score.json', 'w') as f:
        json.dump(score, f)
    with open(ckpt_path + '/scores.json', 'w') as f:
        json.dump(scores, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None,
                    help='ckpt path')
    args = parser.parse_args()
    main(args.ckpt_path)
