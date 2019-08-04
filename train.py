#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:22:56 2018

@author: galo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import torch.optim as optim
import time
from six.moves import cPickle
import opts
import sys
import models
from dataloader import *
from dataloader_conceptual import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer
# import results.plots_general_curve as plots_general_curve
from plots_general_curve import create_dicts_and_json_after_training
import results.html as html
from optimizer import load_optimizer, save_optimizer, zeroing_optimizer, \
    update_optimizer
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def polynomial_decay(epoch, decay_factor, power, initial_rate=1):
    rate = np.minimum(initial_rate, initial_rate * 1/(decay_factor*(1+epoch))
                      **power)
    return rate


def soft_sampling(opt, epoch, model, epoch_start):
    decay_power = 0.5
    if opt.retrieval_reward == 'multinomial_soft':
        model.caption_generator.prob_multinomial_soft = 1 - polynomial_decay(
            epoch - epoch_start, opt.softmax_cooling_decay_factor,
            power=decay_power, initial_rate=1)
        print(f'epoch = {epoch}, prob_multinomial_soft = '
              f'{model.caption_generator.prob_multinomial_soft}')

    elif opt.retrieval_reward == 'gumbel_softmax':
        model.caption_generator.prob_gumbel_softmax = 1 - polynomial_decay(
            epoch - epoch_start, opt.softmax_cooling_decay_factor,
            power=decay_power, initial_rate=1)
        print(f'epoch = {epoch}, prob_gumbel_softmax = '
              f'{model.caption_generator.prob_gumbel_softmax}')



def update_learning_rate(opt, epoch, optimizer_dict, optimizer):
    if epoch > opt.learning_rate_decay_start >= 0:
        frac = (epoch - opt.learning_rate_decay_start) // \
               opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
        if opt.is_alternating:  # if alternating update all optimizers
            if opt.retrieval_reward == 'reinforce':
                for agent in optimizer_dict.keys():
                    # set the decayed rate
                    utils.set_lr(optimizer_dict[agent],
                                 opt.current_lr)
            else:  # gumbel \ multinomial case
                for agent in optimizer_dict.keys():
                    if agent == 'speaker':
                        for agentIn in optimizer_dict['speaker'].keys():
                            # set the decayed rate
                            utils.set_lr(optimizer_dict['speaker'][
                                             agentIn], opt.current_lr)
        elif not opt.is_alternating:  # phase = 1,2 or 3
            # Set the decayed rate
            utils.set_lr(optimizer, opt.current_lr)

        if opt.retrieval_reward == 'reinforce':
            # Set the decayed rate
            utils.set_lr(optimizer, opt.current_lr)
    else:
        opt.current_lr = opt.learning_rate


def scheduled_sampling_prob(epoch, opt, model):
    frac = (epoch - opt.scheduled_sampling_start) // \
           opt.scheduled_sampling_increase_every
    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac,
                      opt.scheduled_sampling_max_prob)
    model.caption_generator.ss_prob = opt.ss_prob


def retrieval_weight_decay(epoch, opt, model):
    frac = (epoch - opt.retrieval_reward_weight_decay_start) // \
           opt.retrieval_reward_weight_decay_every
    model.retrieval_reward_weight = opt.retrieval_reward_weight * (
            opt.retrieval_reward_weight_decay_rate ** frac)


def save_pkl(args, file_name, save_me, iteration=None, best=None):
    """
    There are three possible file extensions:
    if iteration is not None - ***-309000.pkl
    if best is not None - ***-best.pkl
    if both iteration and best are None - ***.pkl
    """
    assert not (iteration is not None and best is not None), \
        'Only one of (iteration, best) can be different than None, no both'
    checkpoint_path = args['checkpoint_path']
    id = args['id']

    if iteration:
        extension_str = f'-{iteration}'
    elif best:
        extension_str = '-best'
    else:
        extension_str = ''
    with open(os.path.join(
            checkpoint_path,
            file_name + '_' + id + extension_str + '.pkl'),
            'wb') as f:
        cPickle.dump(save_me, f)
    print(f'{file_name}_{id}{extension_str}.pkl saved to {checkpoint_path}')


def save_model(model, opt, model_kind, iteration=None):
    checkpoint_path = os.path.join(opt.checkpoint_path, model_kind + '.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f'{model_kind} model saved to {checkpoint_path}')
    if iteration:
        checkpoint_path = os.path.join(
            opt.checkpoint_path, model_kind + '-' + str(iteration) + '.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'{model_kind} model saved to {checkpoint_path}')


def check_equal_embed_weights(model):
    # In case of shared embedding - checks if after every optimizer step
    # the embeddings are the same
    assert ((model.caption_generator.embed[0] is not
         model.vse.txt_enc.embed)
            or (model.caption_generator.embed[0] !=
                model.vse.txt_enc.embed)), 'Error - embedding should be the ' \
                                           'same after each optimizer step ' \
                                           'due to the usage of shared_embed'


def load_infos(opt):
    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        filename = os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')
        with open(filename, 'rb') as f:
            print("read from [%s]" % filename)
            # infos = cPickle.load(f)
            infos = cPickle.load(f, encoding='latin1')
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size",
                            "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                    "Command line argument and saved model disagree on '%s' " \
                    % checkme
    return infos


def load_data(data, opt):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if opt.use_att:
        if data['att_masks'] is not None:
            all_features = [data['fc_feats'], data['att_feats'],
                            data['att_masks'], data['labels'], data['masks']]
        else:  # data['att_masks'] is None
            all_features = [data['fc_feats'], data['att_feats'],
                            data['labels'], data['masks']]
    else:
        all_features = [data['fc_feats'], data['labels'], data['masks']]
    if torch.cuda.is_available():
        all_features = utils.var_wrapper(all_features)
    else:  # CPU
        all_features = utils.var_wrapper(all_features, cuda=False)
    return all_features


def move_model_to_gpu(model):
    if torch.cuda.is_available():
        model.cuda()
    else:  # CPU
        model.cpu()


def load_histories(opt):
    histories = {}
    if opt.start_from is not None:
        if os.path.isfile(
                os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from,
                                   'histories_' + opt.id + '.pkl'), 'rb') as f:
                    histories = cPickle.load(f)
    return histories


def forward_and_backward(model, opt, fc_feats, labels, masks, data,
                         curr_turn, att_feats=None, att_masks=None):
    if opt.is_alternating:  # Alternate training
        # Switch turns for alternating loss calculations
        loss = model(fc_feats, labels, masks, data, att_feats, att_masks,
                     is_alternating=True, alternating_turn=curr_turn)
    else:  # Regular training
        loss = model(fc_feats, labels, masks, data, att_feats, att_masks)
    loss.backward()
    return loss


def print_to_log(model, iteration, epoch, start, end, train_loss):
    np_train_loss = train_loss.cpu().numpy()
    print(f"iter {iteration} (epoch {epoch}), "
          f"train_loss = {np_train_loss :.2f}, "
          # f"train_loss = {round(train_loss.cpu().numpy())}, "
          f"time/batch = {round(end - start, 2)}")
    prt_str = ""
    for k, v in model.loss().items():
        if torch.is_tensor(v):
            v = np.round(v.cpu().numpy(), 2)
        else: # Not a tensor
            v = np.round(v, 2)
            # prt_str += f'{k} = {np.round(v.cpu().numpy(), 2)} '
        # else:  # Not a tensor
        prt_str += f' {k} = ' + str(v)
    print(prt_str)


def update_iteration_and_epoch(iteration, epoch, update_lr_flag, data):
    # Update the iteration and epoch
    iteration += 1
    if data['bounds']['wrapped']:
        epoch += 1
        update_lr_flag = True
    return iteration, epoch, update_lr_flag


def write_loss_summary(iteration, opt, train_loss, model, loss_history,
                       lr_history, ss_prob_history):
    # Write the training loss summary
    loss_history[iteration] = train_loss
    lr_history[iteration] = opt.current_lr
    ss_prob_history[iteration] = model.caption_generator.ss_prob
    return loss_history, lr_history, ss_prob_history


def evaluate_model(opt, model, loader, iteration, val_result_history):
    # eval model
    eval_kwargs = {'split': 'val',
                   'dataset': opt.input_json, 'use_att': opt.use_att}
    eval_kwargs.update(vars(opt))
    # Load the retrieval model for evaluation
    val_loss, predictions, lang_stats = eval_utils.eval_split(
        model, loader, eval_kwargs, useGenSent=opt.rank_on_gen_captions)

    val_result_history[iteration] = {'loss': val_loss,
                                     'lang_stats': lang_stats,
                                     'predictions': predictions}
    return val_result_history, lang_stats, val_loss


def get_current_score(opt, lang_stats, val_loss):
    # Save model if is improving on validation result
    if opt.language_eval == 1:
        # current_score = lang_stats['SPICE']*100
        # Changing from SPICE to CIDEr,
        # because that SPICE isn't implemented for now
        if opt.phase == 1:
            current_score = lang_stats['val']['CIDEr']
        else:
            current_score = lang_stats['CIDEr']
    else:
        # In case we pretraining the listener, there is no loss_cap
        if opt.phase == 1:
            current_score = 0
        else:
            current_score = - val_loss['loss_cap']
    try:
        current_score_vse = val_loss.get(opt.vse_eval_criterion, 0) * 100
    except:
        current_score_vse = val_loss['val'].get(opt.vse_eval_criterion, 0) * 100
    return current_score, current_score_vse


def check_if_best(current_score, best_val_score, current_score_vse,
                  best_val_score_vse):
    best_flag = False
    best_flag_vse = False
    if best_val_score is None or current_score > best_val_score:
        best_val_score = current_score
        best_flag = True
    if best_val_score_vse is None \
            or current_score_vse > best_val_score_vse:
        best_val_score_vse = current_score_vse
        best_flag_vse = True
    return best_val_score, best_flag, best_val_score_vse, best_flag_vse


def save_any_kind_of_model(model, opt, iteration):
    if opt.is_alternating:  # If alternating
        save_model(model, opt, model_kind='alternatingModel',
                   iteration=iteration)

    else:  # Non alternating training
        save_model(model, opt, model_kind='model',
                   iteration=iteration)


def dump_infos_miscellaneous(iteration, epoch, loader, best_val_score,
                             best_val_score_vse, opt, infos, model):
    infos['iter'] = iteration
    infos['epoch'] = epoch
    infos['iterators'] = loader.iterators
    infos['split_ix'] = loader.split_ix
    infos['best_val_score'] = best_val_score
    infos['best_val_score_vse'] = best_val_score_vse
    infos['opt'] = opt
    infos['vocab'] = loader.get_vocab()
    infos['gumbel_temp'] = model.caption_generator.gumbel_temp
    return infos


def dump_histories_miscellaneous(val_result_history, loss_history, lr_history,
                                 ss_prob_history, histories):
    histories['val_result_history'] = val_result_history
    histories['loss_history'] = loss_history
    histories['lr_history'] = lr_history
    histories['ss_prob_history'] = ss_prob_history
    return histories


def save_results(args, iteration, infos, histories):
    save_pkl(args, save_me=infos, file_name='infos')
    save_pkl(args, save_me=infos,
             file_name='infos', iteration=iteration)
    save_pkl(args, save_me=histories, file_name='histories')


def save_best_results(best_flag, best_flag_vse, opt, args, model, infos):
    if best_flag:
        save_model(model, opt, model_kind='model-best')
        save_pkl(args, save_me=infos, file_name='infos', best=True)

    if best_flag_vse:
        save_model(model, opt, model_kind='model_vse-best')
        save_pkl(args, save_me=infos,
                 file_name='infos_vse', best=True)


def load_infos_histories_loader(opt):
    opt.use_att = utils.if_use_att(opt)
    # loader = DataLoader(opt)
    loader = DataLoader_conceptual(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    infos = load_infos(opt)
    histories = load_histories(opt)
    return loader, infos, histories


def load_from_infos(infos, loader, opt):
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    epoch_start = epoch
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    opt.gumbel_temp = infos.get('gumbel_temp', opt.gumbel_temp)
    return epoch, epoch_start, iteration


def load_best_score(opt, infos):
    best_val_score, best_val_score_vse = None, None
    if opt.load_best_score:
        best_val_score = infos.get('best_val_score', None)
        best_val_score_vse = infos.get('best_val_score_vse', None)
    return best_val_score, best_val_score_vse


def load_from_histories(histories):
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    return val_result_history, loss_history, lr_history, ss_prob_history


def set_model_and_optimizer(opt):
    model = models.AlternatingJointModel(opt)
    move_model_to_gpu(model)
    update_lr_flag = True
    if opt.share_embed:  # if using shared embedding
        model.caption_generator.embed[0] = model.vse.txt_enc.embed
    # Assure in training mode
    model.train()
    # creates / load optimizer dictionary, one per alternating turn
    optimizer_dict = load_optimizer(model, opt)
    return model, optimizer_dict, update_lr_flag


def temperature_annealing(opt, epoch, model, iteration, epoch_start=200,
                          iteration_start=177000):

    frac = max(0.5, np.exp(
        -opt.gumbel_temperature_annealing_factor*(
            iteration - iteration_start)))
    model.caption_generator.gumbel_temp = \
        model.caption_generator.gumbel_temp * frac

    # frac = (epoch - epoch_start) // \
    #        opt.gumbel_temperature_annealing_rate
    # model.caption_generator.gumbel_temp = \
    #     opt.gumbel_temp * opt.gumbel_temperature_annealing_factor ** frac
    print(f'annealing_factor (r) = {opt.gumbel_temperature_annealing_factor}, '
          f'tau = {frac}, gumbel_temp = '
          f'{model.caption_generator.gumbel_temp}')

def update_lr_scheduled_sampling_weight_decay(
        update_lr_flag, opt, epoch, optimizer_dict, optimizer, model,
        epoch_start, iteration):
    if update_lr_flag:
        update_learning_rate(opt, epoch, optimizer_dict, optimizer)
        # Assign the scheduled sampling prob
        if epoch > opt.scheduled_sampling_start >= 0:
            scheduled_sampling_prob(epoch, opt, model)
        # Assign retrieval loss weight
        if epoch > opt.retrieval_reward_weight_decay_start >= 0:
            retrieval_weight_decay(epoch, opt, model)
        update_lr_flag = False

    # Scheduling the values of prob_*_soft with softmax_cooling_decay_factor
    if opt.softmax_cooling_decay_factor > 0:
        soft_sampling(opt, epoch, model, epoch_start)
    if opt.gumbel_temperature_annealing_factor > 0 and iteration % \
            opt.num_iteration_for_annealing == 0:
        temperature_annealing(opt, epoch, model, iteration, epoch_start)
    return update_lr_flag


def operations_in_checkpoint(
        opt, model, loader, iteration, epoch, best_val_score,
        best_val_score_vse, optimizer_dict, infos, histories, loss_history,
        lr_history, ss_prob_history, val_result_history):
    # Evaluate model
    val_result_history, lang_stats, val_loss = evaluate_model(
        opt, model, loader, iteration, val_result_history)

    current_score, current_score_vse = get_current_score(
        opt, lang_stats, val_loss)

    best_val_score, best_flag, best_val_score_vse, best_flag_vse = \
        check_if_best(current_score, best_val_score, current_score_vse,
                      best_val_score_vse)

    # Save model
    save_any_kind_of_model(model, opt, iteration)

    # Save optimizers
    save_optimizer(opt, optimizer_dict)

    # Dump miscellaneous information
    infos = dump_infos_miscellaneous(
        iteration, epoch, loader, best_val_score, best_val_score_vse,
        opt, infos, model)
    histories = dump_histories_miscellaneous(
        val_result_history, loss_history, lr_history, ss_prob_history,
        histories)

    # Save results, information and best results
    args = {'checkpoint_path': opt.checkpoint_path, 'id': opt.id}
    save_results(args, iteration, infos, histories)
    save_best_results(best_flag, best_flag_vse, opt, args, model, infos)


def train(opt):
    loader, infos, histories = load_infos_histories_loader(opt)
    epoch, epoch_start, iteration = load_from_infos(infos, loader, opt)
    best_val_score, best_val_score_vse = load_best_score(opt, infos)
    val_result_history, loss_history, lr_history, ss_prob_history = \
        load_from_histories(histories)

    model, optimizer_dict, update_lr_flag = set_model_and_optimizer(opt)

    num_turns = len(opt.alternating_turn) if opt.is_alternating else 1
    init_scorer(opt.cached_tokens)

    while True:
        # Switch turns for alternating loss & optimizers calculations
        curr_turn = opt.alternating_turn[iteration % num_turns] \
            if opt.is_alternating else 'optimizer'

        print('----  Start {} turn  ----'.format(curr_turn))

        # get suitable optimizer
        optimizer = optimizer_dict[curr_turn]
        update_lr_flag = update_lr_scheduled_sampling_weight_decay(
            update_lr_flag, opt, epoch, optimizer_dict, optimizer, model,
            epoch_start, iteration)

        start = time.time()
        data = loader.get_batch('train')
        if opt.use_att:
            if data['att_masks'] is not None:
                fc_feats, att_feats, att_masks, labels, masks = load_data(
                    data, opt)
            # No att_masks since all att_feats as the same size
            else:
                fc_feats, att_feats, labels, masks = load_data(data, opt)
        else:
            fc_feats, labels, masks = load_data(data, opt)
        print('Read data:', time.time() - start)

        start = time.time()
        zeroing_optimizer(opt, optimizer_dict, optimizer)
        if opt.use_att:
            if data['att_masks'] is not None:
                loss = forward_and_backward(
                    model, opt, fc_feats, labels, masks, data, curr_turn,
                    att_feats, att_masks)
            # If all att_feats as the same size
            else:
                loss = forward_and_backward(
                    model, opt, fc_feats, labels, masks, data, curr_turn,
                    att_feats, att_masks=None)
        else:
            loss = forward_and_backward(model, opt, fc_feats, labels, masks,
                                        data, curr_turn, att_feats=None,
                                        att_masks=None)

        update_optimizer(optimizer_dict, optimizer, opt)

        if opt.share_embed:
            check_equal_embed_weights(model)

        train_loss = loss.data[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()

        print_to_log(model, iteration, epoch, start, end, train_loss)

        iteration, epoch, update_lr_flag = update_iteration_and_epoch(
            iteration, epoch, update_lr_flag, data)
        if iteration % opt.losses_log_every == 0 or opt.start_with_checkpoint:
            loss_history, lr_history, ss_prob_history = write_loss_summary(
                iteration, opt, train_loss, model, loss_history, lr_history,
                ss_prob_history)

        # Make evaluation on validation set, and save model
        if iteration % opt.save_checkpoint_every == 0 or \
                opt.start_with_checkpoint:

            operations_in_checkpoint(
                opt, model, loader, iteration, epoch, best_val_score,
                best_val_score_vse, optimizer_dict, infos, histories,
                loss_history, lr_history, ss_prob_history, val_result_history)

        if opt.start_with_checkpoint:
            opt.start_with_checkpoint = 0
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs != -1:
            # Announce that run has been finished
            print("Finished training")
            break


def main():
    opt = opts.parse_opt()
    train(opt)
    # If we in pretrain phase, return without creating dictionaries
    if opt.phase in [1, 2]:
        return
    # Create val and test dictionaries and json file
    # plots_general_curve.create_dicts_and_json_after_training(opt)
    create_dicts_and_json_after_training(opt)
    # Create HTML file for the model
    if opt.dataset == 'coco':
        html.create_html_after_train(opt)
    print("Finished all")


if __name__ == '__main__':
    main()
