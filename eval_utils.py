from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval(dataset, preds, model_id, split, ckpt_path, annFile=None,
                  phase=0):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        #  In case we are using plots_general_curve.py cd is different so
        #  path should be different as well
        if not annFile:
            annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr8k' in dataset:
        annFile = 'coco-caption/annotations/captions_flickr8k.json'
    elif 'flickr30k' in dataset:
            annFile = 'coco-caption/annotations/captions_flickr30k.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    try:
        # coco = COCO(annFile)
        if 'coco' in dataset:
            coco = COCO("/home/lab/vgilad/PycharmProjects/JointImageCaptioning"
                        "/coco-caption/annotations/captions_val2014.json")
        else:  # flickr
            coco = COCO(annFile)
    except:
        coco = COCO(os.path.join('..', annFile))
    valids = coco.getImgIds()
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path,
                               'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, ckpt_path)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    # collect SPICE_sub_score
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in  imgToEval.values()])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()


    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    for i in range(len(preds)):
        if preds[i]['image_id'] in imgToEval:
            preds[i]['eval'] = imgToEval[preds[i]['image_id']]
    # filter results to only those in MSCOCO validation set (will be about a third)
    json.dump(preds, open(
        os.path.join('eval_results/', model_id + '_' + split + '_nofilt.json'),
        'w'))
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model, loader, eval_kwargs={}, annFile=None, useGenSent=False):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    rank_eval = eval_kwargs.get('rank_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    batch_size = eval_kwargs.get('batch_size', 0)
    phase = eval_kwargs.get('phase', 0)
    use_att = eval_kwargs.get('use_att', True)
    ckpt_path = eval_kwargs.get('checkpoint_path', None)
    # Make sure in the evaluation mode
    model.eval()

    np.random.seed(123)
    loader.reset_iterator(split)
    n = 0
    losses = {}
    loss_evals = 1e-8
    predictions = [] # Save the discriminative results. Used for further html visualization.
    with torch.no_grad():
        while True:
            data = loader.get_batch(split)
            n = n + loader.batch_size

            if data.get('labels', None) is not None:
                # forward the model to get loss
                if use_att:
                    if data['att_masks'] is not None:
                        tmp = [data['fc_feats'], data['att_feats'],
                               data['labels'], data['masks'], data['att_masks']]
                    else:
                        tmp = [data['fc_feats'], data['att_feats'],
                               data['labels'], data['masks']]
                else:
                    tmp = [data['fc_feats'], data['labels'], data['masks']]
                if torch.cuda.is_available():
                    if torch.is_tensor(tmp[0]):
                        tmp = [Variable(_, volatile=True).cuda() for _ in tmp]
                    else:  # numpy array
                        tmp = [Variable(torch.from_numpy(_),
                                        volatile=True).cuda() for _ in tmp]
                else:
                    if torch.is_tensor(tmp[0]):
                        tmp = [Variable(_, volatile=True) for _ in tmp]
                    else:  # numpy array
                        tmp = [Variable(torch.from_numpy(_), volatile=True)
                               for _ in tmp]
                if use_att:
                    if data['att_masks'] is not None:
                        fc_feats, att_feats, labels, masks, att_masks = tmp
                        loss = model(fc_feats, labels, masks, data, att_feats,
                                     att_masks)
                    else:  #  In case att_masks is None
                        fc_feats, att_feats, labels, masks = tmp
                        loss = model(fc_feats, labels, masks, data, att_feats,
                                     att_masks=None)
                else:
                    fc_feats, labels, masks = tmp
                    loss = model(fc_feats, labels, masks,
                                 data, att_feats=None, att_masks=None)
                loss = loss.data[0]
                for k, v in model.loss().items():
                    if k not in losses:
                        losses[k] = 0
                    losses[k] += v

                loss_evals = loss_evals + 1


            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            if use_att:
                tmp = [data['fc_feats'][
                           np.arange(loader.batch_size) * loader.seq_per_img],
                       data['att_feats'][
                           np.arange(loader.batch_size) * loader.seq_per_img]]
                if data['att_masks'] is not None:
                       tmp.append(data['att_masks'][
                           np.arange(loader.batch_size) * loader.seq_per_img])
                tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                        volatile=True)
                if data['att_masks'] is not None:
                    fc_feats, att_feats, att_masks = tmp
                else:
                    fc_feats, att_feats = tmp
                    att_masks = None
            else:
                tmp = data['fc_feats'][
                           np.arange(loader.batch_size) * loader.seq_per_img]
                tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                        volatile=True)
                fc_feats = tmp
                att_feats, att_masks = None, None

            # forward the model to also get generated samples for each image
            seq, _ = model.sample(fc_feats, att_feats, att_masks,
                                  opt=eval_kwargs)

            sents = utils.decode_sequence(loader.get_vocab(), seq.data)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                if eval_kwargs.get('dump_path', 0) == 1:
                    entry['file_name'] = data['infos'][k]['file_path']
                predictions.append(entry)
                if eval_kwargs.get('dump_images', 0) == 1:
                    # dump the raw image to vis/ folder
                    cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                                data['infos'][k][
                                                    'file_path']) + '" vis/imgs/img' + str(
                        len(predictions)) + '.jpg'  # bit gross
                    print(cmd)
                    os.system(cmd)

                if verbose:
                    print(
                        'image %s: %s' % (entry['image_id'], entry['caption']))

            # if we wrapped around the split or used up val imgs budget then bail
            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if num_images != -1:
                ix1 = min(ix1, num_images)
            for i in range(n - ix1):
                predictions.pop()

            if verbose:
                print('evaluating validation preformance... %d/%d (%f)' % (
                ix0 - 1, ix1, loss))

            if data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break


    lang_stats = None
    if lang_eval == 1:
        if phase == 1:
            lang_stats = {}
            for split_lang in ['val', 'test']:
                lang_stats[split_lang] = language_eval(dataset, predictions,
                                                       eval_kwargs['id'],
                                                       split_lang,
                                                       annFile, phase)
        else:
            lang_stats = language_eval(dataset, predictions, eval_kwargs['id'],
                                   split, ckpt_path, annFile, phase)

    else:
        lang_stats = {}

    if useGenSent:
        # rank generated captions
        if rank_eval:
            ranks = evalrank(model, loader, eval_kwargs, useGenSent)
        else:
            ranks = {}
        #  annFile will not be None only if we run eval.py, and then,
        #  we don't want to have gt_ranks
        #  rank gt captions
        if rank_eval and not annFile:
            gt_ranks = evalrank(model, loader, eval_kwargs, False)
        else:
            gt_ranks = {}
    else:  # useGenSent - False
        # rank gt captions
        if rank_eval:
            if phase == 1:
                ranks = {}
                old_split = eval_kwargs.get('split')
                for split_rank in ['val', 'test']:
                    eval_kwargs['split'] = split_rank
                    ranks[split_rank] = evalrank(model, loader, eval_kwargs,
                                                   useGenSent)
                eval_kwargs['split'] = old_split
            else:
                ranks = evalrank(model, loader, eval_kwargs, useGenSent)
        else:
            ranks = {}

    # Switch back to training mode
    model.train()
    losses = {k: v / loss_evals for k, v in losses.items()}
    losses.update(ranks)
    if useGenSent and not annFile:
        losses['gt_ranks'] = gt_ranks

    return losses, predictions, lang_stats


def encode_data(model, loader, eval_kwargs={}, useGenSent=False):
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    # Use ground truth captions (5 per image), In conceptual captions there
    # is one caption per image
    if not useGenSent and loader.dataset in ['coco', 'flickr8k', 'flickr30k']:
        loader.seq_per_img = 5
    # use generated sentences (1 per image) or conceptual captions
    else:
        loader.seq_per_img = 1
    loader.reset_iterator(split)

    n = 0
    img_embs = []
    cap_embs = []
    images_data = []  # save image data such as image id, list of tuples
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        if not (useGenSent):  # Use ground truth captions
            tmp = [data['fc_feats'], data['att_feats'], data['labels'],
                   data['masks']]
            tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                    volatile=True)
            fc_feats, att_feats, labels, masks = tmp

            img_emb = model.vse.img_enc(fc_feats)
            cap_emb = model.vse.txt_enc(labels, masks)

        else:  # Use generated sentences
            '''
            Make a generted sampled sentence per image
            '''
            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp = [data['fc_feats'][
                       np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][
                       np.arange(loader.batch_size) * loader.seq_per_img]]
            if data['att_masks'] is not None:
                   tmp.append(data['att_masks'][
                       np.arange(loader.batch_size) * loader.seq_per_img])

            tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                    volatile=True)
            if data['att_masks'] is not None:
                fc_feats, att_feats, att_masks = tmp
            else:
                fc_feats, att_feats = tmp
                att_masks = data['att_masks']
            # forward the model to also get generated samples for each image
            # seq, _ = model.sample(fc_feats, att_feats, att_masks, opt=eval_kwargs)
            # sample using argmax
            if data['att_masks'] is not None:
                seq, _ = model.caption_generator.sample(*utils.var_wrapper([
                    fc_feats, att_feats, att_masks],
                cuda=torch.cuda.is_available(),
                    volatile=True), opt={'sample_max': 1, 'temperature': 1})
            else:
                seq, _ = model.caption_generator.sample(*utils.var_wrapper([
                    fc_feats, att_feats],
                    cuda=torch.cuda.is_available(), volatile=True), att_masks,
                    opt={'sample_max': 1, 'temperature': 1})
            # DEBUG:
            #            sents = utils.decode_sequence(loader.get_vocab(), seq.data)
            #            print('The generated sentences for the encoding data are : \n {}'.format(sents))
            # DEBUG end
            img_emb = model.vse.img_enc(fc_feats)
            # using generated sentence instead of labels
            beginningOfSenChar = (torch.ones(seq.shape[0]) * (
                        len(loader.get_vocab().keys()) + 1)).long().view(-1, 1)
            seq_masks = torch.cat(
                [Variable(seq.data.new(seq.size(0), 2).fill_(1).float()),
                 (seq > 0).float()[:, :-1]], 1)
            seq_masks = utils.var_wrapper(seq_masks,
                                          cuda=torch.cuda.is_available(),
                                          volatile=True)
            # adding the beginning of sentence char
            if torch.cuda.is_available():
                seq = torch.from_numpy(np.hstack(
                    [beginningOfSenChar.cpu().data.numpy(),
                     seq.cpu().data.numpy()])).cuda()
            else:  # CPU()
                seq = torch.from_numpy(np.hstack(
                    [beginningOfSenChar.cpu().data.numpy(),
                     seq.cpu().data.numpy()]))
            try:
                cap_emb = model.vse.txt_enc(seq, seq_masks)
            except (RuntimeError) as e:
                print('\n Reached an exception, continute... \n')
                print(e)
                continue

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)

        if n > ix1:
            img_emb = img_emb[:(ix1 - n) * loader.seq_per_img]
            cap_emb = cap_emb[:(ix1 - n) * loader.seq_per_img]
            images_data += data['infos'][:(
                        ix1 - n)]  # save only the necessary images id from the batch
        else:
            images_data += data['infos']  # save all batch images id

        # preserve the embeddings by copying from gpu and converting to np
        img_embs.append(img_emb.data.cpu().numpy().copy())
        cap_embs.append(cap_emb.data.cpu().numpy().copy())

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        print("%d/%d" % (n, ix1))

    img_embs = np.vstack(img_embs)
    cap_embs = np.vstack(cap_embs)

    assert img_embs.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs, images_data


def evalrank(model, loader, eval_kwargs={}, useGenSent=False):
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    fold5 = eval_kwargs.get('fold5', 0)
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    if not (useGenSent):
        print('Computing results useGenSent = False...')
        img_embs, cap_embs, images_data = encode_data(model, loader,
                                                      eval_kwargs)
        if loader.dataset in ['coco', 'flickr8k', 'flickr30k']:
            img_shape = img_embs.shape[0] / 5
        elif loader.dataset == 'conceptual':
            img_shape = img_embs.shape[0]
        print('Images: %d, Captions: %d' %
              (img_shape, cap_embs.shape[0]))

        if not fold5:
            # no cross-validation, full evaluation
            r, rt = i2t(img_embs, cap_embs, measure='cosine', return_ranks=True)
            ri, rti, images_ranking = t2i(img_embs, cap_embs, images_data,
                                          measure='cosine', return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)  # sum of all recalls (3*i2t + 3*t2i )
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                             cap_embs[i * 5000:(i + 1) *
                                               5000], measure='cosine',
                             return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                               cap_embs[i * 5000:(i + 1) *
                                                 5000], measure='cosine',
                               return_ranks=True)
                if i == 0:
                    rt, rti = rt0, rti0
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])

        return {'rsum': rsum, 'i2t_ar': ar, 't2i_ar': ari,
                'i2t_r1': r[0], 'i2t_r5': r[1], 'i2t_r10': r[2],
                'i2t_medr': r[3], 'i2t_meanr': r[4],
                't2i_r1': ri[0], 't2i_r5': ri[1], 't2i_r10': ri[2],
                't2i_medr': ri[3], 't2i_meanr': ri[4],
                'gt_images_ranking': images_ranking}  # {'rt': rt, 'rti': rti}
    else:  # use generated sentence
        print('Computing results for generated samples...')
        img_embs, cap_embs, images_data = encode_data(model, loader,
                                                      eval_kwargs, useGenSent)
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0], cap_embs.shape[0]))

        if not fold5:
            # no cross-validation, full evaluation
            ri, rti, images_ranking = t2i(img_embs, cap_embs, images_data,
                                          measure='cosine', return_ranks=True,
                                          useGenSent=useGenSent)
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)  # sum of all recalls (3*t2i )
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                             cap_embs[i * 5000:(i + 1) *
                                               5000], measure='cosine',
                             return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                               cap_embs[i * 5000:(i + 1) *
                                                 5000], measure='cosine',
                               return_ranks=True)
                if i == 0:
                    rt, rti = rt0, rti0
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])

        return {'rsum': rsum, 't2i_ar': ari,
                't2i_r1': ri[0], 't2i_r5': ri[1], 't2i_r10': ri[2], \
                't2i_medr': ri[3], 't2i_meanr': ri[4],
                'images_ranking': images_ranking}


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, images_data, npts=None, measure='cosine',
        return_ranks=False, useGenSent=False):
    """
    Text->Images (Image Search)
    in case useGenSent=False - using ground truth caption and then:
        Images: (5N, K) matrix of images
        Captions: (5N, K) matrix of captions
    else: (using one generated caption per image)
        Images: (N, K) matrix of images
        Captions: (N, K) matrix of captions
    """
    images_ranking = {}
    images_ranking = {}
    if not (useGenSent):  # 5 ground truth captions per image
        numberOfCaptionsPerImage = 5
    else:  # 1 generated caption per image
        numberOfCaptionsPerImage = 1
    if npts is None:
        npts = images.shape[0] // numberOfCaptionsPerImage
    ims = np.array(
        [images[i] for i in range(0, len(images), numberOfCaptionsPerImage)])

    ranks = np.zeros(numberOfCaptionsPerImage * npts)
    top1 = np.zeros(numberOfCaptionsPerImage * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[
                  numberOfCaptionsPerImage * index:numberOfCaptionsPerImage * index + numberOfCaptionsPerImage]  # get <numberOfCaptionsPerImage> captions - 1 or 5

        # Compute scores
        if measure == 'order':
            bs = 100
            if numberOfCaptionsPerImage * index % bs == 0:
                mx = min(captions.shape[0],
                         numberOfCaptionsPerImage * index + bs)
                q2 = captions[numberOfCaptionsPerImage * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (numberOfCaptionsPerImage * index) % bs:(numberOfCaptionsPerImage * index) % bs + numberOfCaptionsPerImage].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(
                len(inds)):  # index is for image; i is for caption (1 or 5)
            '''
            sort each line (i) wich represent caption i (1 of 5 or just 1), 
            in decresing order of dot product with all the indexes of the images
            '''
            inds[i] = np.argsort(d[i])[
                      ::-1]  # sort scores of all images from high to low
            ranks[numberOfCaptionsPerImage * index + i] = \
            np.where(inds[i] == index)[0][0]
            '''
            gives a data structure of the following:
                in case of 5 captions: 
                    - 5 first cells are rankes of image 0 for suitable capions 1...5, 
                    - the next 5 cells are rankes of image 1 for suitable caption 1..5, etc...
                in case of 1 caption:
                    - every index is the rank of the suitable image (at 0 the rank of image 0, at 1 the rank of image 1 etc..)
            '''
            top1[numberOfCaptionsPerImage * index + i] = inds[i][
                0]  # the best image

            rank_i = ranks[numberOfCaptionsPerImage * index + i] = \
            np.where(inds[i] == index)[0][0]  # rank of correct image

            # images_ranking.update({images_data[index]['id'] : {'rank_correct_im':rank_i}})
            if useGenSent:
                images_ranking.update({index: {
                    'image_id': images_data[index]['id'],
                    'rank_correct_im': rank_i,
                    'file_path': images_data[index]['file_path']}})
            else:  # ground truth sentences
                caption_str = 'caption' + str(i)
                if not images_ranking.get(
                        index):  # for 1st ground truth caption for the image
                    images_ranking.update({index: {
                        caption_str: {'image_id': images_data[index]['id'],
                                      'rank_correct_im': rank_i,
                                      'file_path': images_data[index][
                                          'file_path']}}})
                else:  # for other ground truth captions of the image
                    images_ranking[index].update(
                        {caption_str: {'image_id': images_data[index]['id'],
                                       'rank_correct_im': rank_i,
                                       'file_path': images_data[index][
                                           'file_path']}})

            for j in range(4):  # distractors loop
                im_id_rank = 'im_id_rank_' + str(j)
                im_id_rank_url = 'im_url_rank_' + str(j)
                # images_ranking[images_data[index]['id']].update({im_id_rank: images_data[int(inds[0,j])]['id']})
                if useGenSent:
                    images_ranking[index].update(
                        {im_id_rank: images_data[int(inds[0, j])]['id'],
                         im_id_rank_url: images_data[int(inds[0, j])][
                             'file_path']})
                else:  # ground truth sentences
                    images_ranking[index][caption_str].update(
                        {im_id_rank: images_data[int(inds[i, j])]['id'],
                         im_id_rank_url: images_data[int(inds[i, j])][
                             'file_path']})

    # Compute metrics - normalize itself - both in cases of 5 & 1 caption per image
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(
        ranks)  # percentage of first hit ranks
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(
        ranks)  # percentage of hit at top 5
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(
        ranks)  # percentage of hit at top 10
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if (useGenSent):
        print('\n validation rank stats for generated captions: \n r1 {} \n r5\
        {} \n r10 {} \n medr {} \n meanr {} \n \n'
              .format(r1, r5, r10, medr, meanr))
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), images_ranking
    else:
        return (r1, r5, r10, medr, meanr)


def evalrankcap(model, loader, eval_kwargs={}, useGenSent=False):
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    fold5 = eval_kwargs.get('fold5', 0)
    divide_caption = eval_kwargs.get('divide_caption', 0)
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    if not divide_caption:
        # use generated sentence
        print('Computing results for generated samples...')
        # img_embs, cap_embs, images_data = encode_data(model, loader,
        #                                               eval_kwargs, useGenSent)

        img_embs, cap_embs_gt, cap_embs_generated, images_data = \
            encode_data_cap(model, loader, eval_kwargs, useGenSent)
        print('Generated captions: %d, Captions: %d' %
              (cap_embs_generated.shape[0], cap_embs_gt.shape[0]))

        # no cross-validation, full evaluation
        #  Machine generated to ground truth
        m2gt = gen2gt(cap_embs_generated, cap_embs_gt, images_data,
                    measure='cosine', return_ranks=False)
        print("Generated captions to ground truth captions retrieval:\n "
              "r1 {} \n r5 {} \n r10 {} \n median {} \n mean {}"
              .format(m2gt[0], m2gt[1], m2gt[2], m2gt[3], m2gt[4]))

        # print("rsum: %.1f" % rsum)  # sum of all recalls (3*t2i )
        #  Ground truth to machine generated
        gt2m = gt2gen(cap_embs_generated, cap_embs_gt,
                                         images_data, measure='cosine',
                                         return_ranks=False)

        print("Ground truth captions to generated captions retrieval: \n"
              "r1 {} \n r5 {} \n r10 {} \n median {} \n mean {}"
              .format(gt2m[0], gt2m[1], gt2m[2], gt2m[3], gt2m[4]))


        return {'gen2gt_r1': m2gt[0], 'gen2gt_r5': m2gt[1], 'gen2gt_r10': m2gt[2], \
                'gen2gt_medr': m2gt[3], 'gen2gt_meanr': m2gt[4],
                'gt2gen_r1': gt2m[0], 'gt2gen_r5': gt2m[1], 'tgt2gen_r10': gt2m[2],
                'gt2gen_medr': gt2m[3], 'gt2gen_meanr': gt2m[4]}

    else:  # Divide caption
        # use generated sentence
        print('Computing results for generated samples...')
        img_embs, cap_emb_gt_first_half, cap_emb_generated_first_half, \
        cap_emb_gt_second_half, cap_emb_generated_second_half, images_data = \
            encode_data_halves(model, loader, eval_kwargs, useGenSent)

        print('Generated captions: %d, Ground truth captions: %d' %
              (cap_emb_generated_first_half.shape[0],
               cap_emb_gt_first_half.shape[0]))
        halves = ['first', 'second']
        for half in halves:
            if half == 'first':
                cap_emb_generated = cap_emb_generated_first_half
                cap_emb_gt = cap_emb_gt_first_half
            elif half == 'second':
                cap_emb_generated = cap_emb_generated_second_half
                cap_emb_gt = cap_emb_gt_second_half
            # no cross-validation, full evaluation
            #  Machine generated to ground truth
            m2gt = gen2gt(cap_emb_generated, cap_emb_gt,
                          images_data, measure='cosine', return_ranks=False)
            print("{} half caption - generated captions to ground truth captions"
                  " "
                  "retrieval:\n " "r1 {} \n r5 {} \n r10 {} \n median {} \n mean {}"
                  .format(half, m2gt[0], m2gt[1], m2gt[2], m2gt[3], m2gt[4]))

            # print("rsum: %.1f" % rsum)  # sum of all recalls (3*t2i )
            #  Ground truth to machine generated
            gt2m = gt2gen(cap_emb_generated, cap_emb_gt,
                          images_data, measure='cosine',
                          return_ranks=False)

            print("{} half caption - Ground truth captions to generated "
                  "captions retrieval: \n"
                  "r1 {} \n r5 {} \n r10 {} \n median {} \n mean {}"
                  .format(half, gt2m[0], gt2m[1], gt2m[2], gt2m[3], gt2m[4]))
            if half == 'first':
                divide_dict = {'first_half': {'gen2gt_r1': m2gt[0],
                                              'gen2gt_r5': m2gt[1],
                                              'gen2gt_r10': m2gt[2],
                                              'gen2gt_medr': m2gt[3],
                                              'gen2gt_meanr': m2gt[4],
                                              'gt2gen_r1': gt2m[0],
                                              'gt2gen_r5': gt2m[1],
                                              'tgt2gen_r10': gt2m[2],
                                              'gt2gen_medr': gt2m[3],
                                              'gt2gen_meanr': gt2m[4]}}
            elif half == 'second':
                divide_dict['second_half'] = {'gen2gt_r1': m2gt[0],
                                              'gen2gt_r5': m2gt[1],
                                              'gen2gt_r10': m2gt[2],
                                              'gen2gt_medr': m2gt[3],
                                              'gen2gt_meanr': m2gt[4],
                                              'gt2gen_r1': gt2m[0],
                                              'gt2gen_r5': gt2m[1],
                                              'tgt2gen_r10': gt2m[2],
                                              'gt2gen_medr': gt2m[3],
                                              'gt2gen_meanr': gt2m[4]}

        return divide_dict



def encode_data_cap(model, loader, eval_kwargs={}, useGenSent=False):
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    if not (useGenSent):  # Use ground truth captions (5 per image)
        loader.seq_per_img = 5
    else:  # use generated sentences (1 per image)
        loader.seq_per_img = 1
    loader.reset_iterator(split)

    n = 0
    img_embs = []
    cap_embs = []
    cap_embs_gt = []
    cap_embs_generated = []
    images_data = []  # save image data such as image id, list of tuples
    with torch.no_grad():
        while True:
            loader.seq_per_img = 5
            data = loader.get_batch(split)
            n = n + loader.batch_size

            tmp = [data['fc_feats'], data['att_feats'], data['labels'],
                   data['masks']]
            tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                    volatile=True)
            fc_feats, att_feats, labels, masks = tmp

            img_emb = model.vse.img_enc(fc_feats)
            cap_emb_gt = model.cap.txt_enc_generated(labels, masks)

            '''
            Make a generated sampled sentence per image
            '''
            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp = [data['fc_feats'][
                       np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][
                       np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][
                       np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                    volatile=True)
            fc_feats, att_feats, att_masks = tmp
            # forward the model to also get generated samples for each image
            # seq, _ = model.sample(fc_feats, att_feats, att_masks, opt=eval_kwargs)
            # sample using argmax
            seq, _ = model.caption_generator.sample(*utils.var_wrapper([
                fc_feats, att_feats, att_masks], cuda=torch.cuda.is_available(),
                volatile=True), opt={'sample_max': 1, 'temperature': 1})

            # img_emb = model.vse.img_enc(fc_feats)
            # using generated sentence instead of labels
            beginningOfSenChar = (torch.ones(seq.shape[0]) * (
                    len(loader.get_vocab().keys()) + 1)).long().view(-1, 1)
            seq_masks = torch.cat(
                [Variable(seq.data.new(seq.size(0), 2).fill_(1).float()),
                 (seq > 0).float()[:, :-1]], 1)
            seq_masks = utils.var_wrapper(seq_masks,
                                          cuda=torch.cuda.is_available(),
                                          volatile=True)
            # adding the beginning of sentence char
            if torch.cuda.is_available():
                seq = torch.from_numpy(np.hstack(
                    [beginningOfSenChar.cpu().data.numpy(),
                     seq.cpu().data.numpy()])).cuda()
            else:  # CPU()
                seq = torch.from_numpy(np.hstack(
                    [beginningOfSenChar.cpu().data.numpy(),
                     seq.cpu().data.numpy()]))
            try:
                cap_emb_generated = model.cap.txt_enc_generated(seq, seq_masks)
            except (RuntimeError) as e:
                print('\n Reached an exception, continute... \n')
                print(e)
                continue

            # if we wrapped around the split or used up val imgs budget then bail
            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if num_images != -1:
                ix1 = min(ix1, num_images)

            if n > ix1:
                img_emb = img_emb[:(ix1 - n) * loader.seq_per_img]
                cap_emb_gt = cap_emb_gt[:(ix1 - n) * loader.seq_per_img]
                cap_emb_generated = cap_emb_generated[:(ix1 - n)]
                images_data += data['infos'][:(
                        ix1 - n)]  # save only the necessary images id from the batch
            else:
                images_data += data['infos']  # save all batch images id

            # preserve the embeddings by copying from gpu and converting to np
            img_embs.append(img_emb.data.cpu().numpy().copy())
            cap_embs_gt.append(cap_emb_gt.data.cpu().numpy().copy())
            cap_embs_generated.append(cap_emb_generated.data.cpu().numpy().copy())

            if data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break

            print("%d/%d" % (n, ix1))

    # img_embs = np.vstack(img_embs)
    cap_embs_gt = np.vstack(cap_embs_gt)
    cap_embs_generated = np.vstack(cap_embs_generated)

    # assert img_embs.shape[0] == ix1 * loader.seq_per_img
    assert cap_embs_generated.shape[0] == ix1
    assert cap_embs_gt.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs_gt, cap_embs_generated, images_data


# def t2t(images, captions, npts=None, measure='cosine', return_ranks=False):


def gen2gt(cap_embs_generated, cap_embs_gt, npts=None, measure='cosine',
           return_ranks=True):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = None
    if npts is None:
        # npts = images.shape[0] // 5
        npts = cap_embs_generated.shape[0]
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        # im = images[5 * index].reshape(1, images.shape[1])
        cap = cap_embs_generated[index].reshape(1, cap_embs_generated.shape[1])

        # Compute scores
        if measure == 'cosine':
            d = np.dot(cap, cap_embs_gt.T).flatten()

        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def gt2gen(cap_embs_generated, cap_embs_gt, npts=None, measure='cosine',
           return_ranks=True):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    cap_embs_generated = np.repeat(cap_embs_generated, [5], axis=0)

    npts = None
    if npts is None:
        npts = cap_embs_generated.shape[0] // 5
    gen_cap = np.array(
        [cap_embs_generated[i] for i in range(0, len(cap_embs_generated), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = cap_embs_gt[5 * index:5 * index + 5]

        # Compute scores
        d = np.dot(queries, gen_cap.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def encode_data_halves(model, loader, eval_kwargs={}, useGenSent=False):
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    batch_size = eval_kwargs.get('batch_size', 128)
    divide_by_second_half = eval_kwargs.get('divide_by_second_half', 0)
    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    if not (useGenSent):  # Use ground truth captions (5 per image)
        loader.seq_per_img = 5
    else:  # use generated sentences (1 per image)
        loader.seq_per_img = 1
    loader.reset_iterator(split)

    n = 0
    img_embs = []
    cap_embs = []
    cap_embs_gt_first_half = []
    cap_embs_gt_second_half = []
    cap_embs_generated_first_half = []
    cap_embs_generated_second_half = []
    images_data = []  # save image data such as image id, list of tuples
    with torch.no_grad():
        while True:
            loader.seq_per_img = 5
            data = loader.get_batch(split)
            n = n + loader.batch_size

            tmp = [data['fc_feats'], data['att_feats'], data['labels'],
                   data['masks']]
            tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                    volatile=True)
            fc_feats, att_feats, labels, masks = tmp

            img_emb = model.vse.img_enc(fc_feats)
            # cap_emb_gt = model.cap.txt_enc_generated(labels, masks)

            '''
            Make a generated sampled sentence per image
            '''
            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp = [data['fc_feats'][
                       np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][
                       np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][
                       np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = utils.var_wrapper(tmp, cuda=torch.cuda.is_available(),
                                    volatile=True)
            fc_feats, att_feats, att_masks = tmp
            # forward the model to also get generated samples for each image
            # seq, _ = model.sample(fc_feats, att_feats, att_masks, opt=eval_kwargs)
            # sample using argmax
            seq, _ = model.caption_generator.sample(*utils.var_wrapper([
                fc_feats, att_feats, att_masks], cuda=torch.cuda.is_available(),
                volatile=True), opt={'sample_max': 1, 'temperature': 1})

            # img_emb = model.vse.img_enc(fc_feats)
            # using generated sentence instead of labels
            beginningOfSenChar = (torch.ones(seq.shape[0]) * (
                    len(loader.get_vocab().keys()) + 1)).long().view(-1, 1)
            seq_masks = torch.cat(
                [Variable(seq.data.new(seq.size(0), 2).fill_(1).float()),
                 (seq > 0).float()[:, :-1]], 1)
            seq_masks = utils.var_wrapper(seq_masks,
                                          cuda=torch.cuda.is_available(),
                                          volatile=True)
            # adding the beginning of sentence char
            if torch.cuda.is_available():
                seq = torch.from_numpy(np.hstack(
                    [beginningOfSenChar.cpu().data.numpy(),
                     seq.cpu().data.numpy()])).cuda()
            else:  # CPU()
                seq = torch.from_numpy(np.hstack(
                    [beginningOfSenChar.cpu().data.numpy(),
                     seq.cpu().data.numpy()]))

            # Split halves labels,masks, seq, seq_masks
            len_half_caption_gt = torch.floor(
                sum((torch.sum(masks, 1))) / (batch_size * loader.seq_per_img)
                / 2)
            shortest_caption_gt = min(torch.sum(masks, 1))

            len_half_caption_generated = torch.floor(
                sum((torch.sum(seq_masks, 1))) / batch_size / 2)
            shortest_caption_generated = min(torch.sum(seq_masks, 1))
            # Check which batch group has the shortest caption
            # generated or human caption
            if shortest_caption_generated < shortest_caption_gt:
                shortest_caption = shortest_caption_generated
                len_half_caption = len_half_caption_generated
            else:
                shortest_caption = shortest_caption_gt
                len_half_caption = len_half_caption_gt
            # Check if the shorter caption is longer then the length of average
            # caption
            if len_half_caption < shortest_caption:
                half = int(len_half_caption.cpu().numpy())
            else:
                half = int((shortest_caption - 1).cpu().numpy())

            # # First half
            # cap_emb_gt_first_half = model.cap.txt_enc_generated(labels[:, :half],
            #                                          masks[:, :half])
            # cap_emb_generated_first_half = model.cap.txt_enc_generated(seq[:, :half],
            #                                                 seq_masks[:, :half])
            #
            # # Second half
            # cap_emb_gt_second_half = model.cap.txt_enc_gt(labels[:, :half],
            #                                   masks[:, :half])
            # cap_emb_generated_second_half = model.cap.txt_enc_gt(seq[:, :half],
            #                                          seq_masks[:, :half])


            if divide_by_second_half:
                # Make sure that the second half is equal between generated and
                # ground truth pairs
                # Ground truth calculate from which token to split to make the
                # length of the second part of the ground truth caption the same as
                # the second part of the corresponding generated caption
                gt_len = torch.sum(masks, dim=1).cpu().numpy() - 2
                half_gt = (gt_len - half)

                # Same for the generated caption
                generated_len = torch.sum(seq_masks, dim=1).cpu().numpy() - 1
                half_generated = (generated_len - half)

                second_caption_length = int(generated_len[0] - half_generated[0])

                # Create new tensors for first and second splits
                seq_second = torch.zeros(batch_size * loader.seq_per_img,
                                         second_caption_length)
                masks_second = torch.zeros(batch_size * loader.seq_per_img,
                                           second_caption_length)
                gen_seqs_second = torch.zeros(batch_size, second_caption_length)
                gen_masks_second = torch.zeros(batch_size,
                                               second_caption_length)

                # seq_first = torch.zeros(self.batch_size, half)
                # masks_first = torch.zeros(self.batch_size, half)
                # gen_seqs_first = torch.zeros(self.batch_size, half)
                # gen_masks_first = torch.zeros(self.batch_size, half)

                for i in range(batch_size*loader.seq_per_img):
                    seq_second[i, :] = labels[i, int(half_gt[i]):int(gt_len[i])]
                    masks_second[i, :] = masks[i,
                                         int(half_gt[i]):int(gt_len[i])]
                    if i < batch_size:
                        gen_seqs_second[i, :] = seq[i,
                                                int(half_generated[i]): int(
                                                    generated_len[i])]
                        gen_masks_second[i, :] = seq_masks[i, int(
                            half_generated[i]):int(generated_len[i])]

                seq_second = seq_second.type(torch.LongTensor).cuda()
                masks_second = masks_second.type(torch.LongTensor).cuda()
                gen_seqs_second = gen_seqs_second.type(torch.LongTensor).cuda()
                gen_masks_second = gen_masks_second .type(torch.LongTensor).cuda()





            else:  # divide by first part of the caption - equal for all
                # captions

                seq_second = labels[:, half:].type(torch.LongTensor).cuda()
                masks_second = masks[:, half:].type(torch.LongTensor).cuda()
                gen_seqs_second = seq[:, half:].type(torch.LongTensor).cuda()
                gen_masks_second = seq_masks[:, half:].type(torch.LongTensor).cuda()

            # Both divide or not, use this split for the first half of the
            # caption
            seq_first = labels[:, :half].type(torch.LongTensor).cuda()
            masks_first = masks[:, :half].type(torch.LongTensor).cuda()
            gen_seqs_first = seq[:, :half].type(torch.LongTensor).cuda()
            gen_masks_first = seq_masks[:, :half].type(torch.LongTensor).cuda()



            # For first half of the caption
            cap_emb_gt_first_half = model.cap.txt_enc_generated(seq_first,
                                                            masks_first)
            cap_emb_generated_first_half = model.cap.txt_enc_generated(
                gen_seqs_first, gen_masks_first)

            # For second half of the caption
            cap_emb_gt_second_half = model.cap.txt_enc_gt(seq_second, masks_second)
            cap_emb_generated_second_half = model.cap.txt_enc_gt(
                gen_seqs_second, gen_masks_second)






            # if we wrapped around the split or used up val imgs budget then bail
            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if num_images != -1:
                ix1 = min(ix1, num_images)

            if n > ix1:
                img_emb = img_emb[:(ix1 - n) * loader.seq_per_img]
                # First half
                cap_emb_gt_first_half = cap_emb_gt_first_half[
                                        :(ix1 - n) * loader.seq_per_img]
                cap_emb_generated_first_half = cap_emb_generated_first_half[
                                               :(ix1 - n)]
                # Second half
                cap_emb_gt_second_half = cap_emb_gt_second_half[
                                        :(ix1 - n) * loader.seq_per_img]
                cap_emb_generated_second_half = cap_emb_generated_second_half[
                                               :(ix1 - n)]
                images_data += data['infos'][:(
                        ix1 - n)]  # save only the necessary images id from the batch
            else:
                images_data += data['infos']  # save all batch images id

            # preserve the embeddings by copying from gpu and converting to np
            img_embs.append(img_emb.data.cpu().numpy().copy())
            # First half
            cap_embs_gt_first_half.append(
                cap_emb_gt_first_half.data.cpu().numpy().copy())
            cap_embs_generated_first_half.append(cap_emb_generated_first_half.data.cpu().numpy().copy())
            # Second half
            cap_embs_gt_second_half.append(
                cap_emb_gt_second_half.data.cpu().numpy().copy())
            cap_embs_generated_second_half.append(
                cap_emb_generated_second_half.data.cpu().numpy().copy())

            if data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break

            print("%d/%d" % (n, ix1))

    # img_embs = np.vstack(img_embs)
        # First half
        cap_embs_gt_first_half = np.vstack(cap_embs_gt_first_half)
        cap_embs_generated_first_half = np.vstack(cap_embs_generated_first_half)
        # Second half
        cap_embs_gt_second_half = np.vstack(cap_embs_gt_second_half)
        cap_embs_generated_second_half = np.vstack(cap_embs_generated_second_half)

    # assert img_embs.shape[0] == ix1 * loader.seq_per_img
    assert cap_embs_generated_first_half.shape[0] == ix1
    assert cap_embs_gt_first_half.shape[0] == ix1 * loader.seq_per_img
    assert cap_embs_generated_second_half.shape[0] == ix1
    assert cap_embs_gt_second_half.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs_gt_first_half, cap_embs_generated_first_half, \
           cap_embs_gt_second_half, cap_embs_generated_second_half, images_data
