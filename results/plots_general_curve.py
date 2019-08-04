from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib
if "DISPLAY" not in os.environ:
        #raise ValueError('Gal: DISPLAY not in os.environ')
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
        
import argparse
import re
from six.moves import cPickle
import numpy as np
import pandas as pd
from copy import deepcopy
import sys
import json

sys.path.append('/home/lab/vgilad/PycharmProjects/JointImageCaptioning')

import eval

import eval_utils
from os import stat
from pwd import getpwuid

"""
input  
1. add argument to input_model_dir_# following number to the exist 
arguments, to add models for the plots
2. best_by var can be changed to decided which iteration results 
to take (t2i_r10'\cider\bleu4)

output
1. plot requested models
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def remove_unnecessary_models(models, dir, specific_model_dir):
    #  Create list of iterations that we would like to save (models of those
    # iterations), according to iterations were cider, bleu4 and r10 were the
    # best
    ignore = []
    ignore_dict = {}
    for metric, index in models[dir]['best_iteration_by'].items():
        # In case that more than one metric gave the best result in the same
        # iteration, save in ignore only once
        ignore_dict.update({metric: models[dir]['iteration'][index]})
        if models[dir]['iteration'][index] not in ignore:
            ignore.append(models[dir]['iteration'][index])
            # Run over all files in model directory and remove all models
            # beside the
    # models that gave best results
    for file in os.listdir(specific_model_dir):
        if file.startswith("alternatingModel-") \
                or (file.startswith('model-') and file[6].isdigit()):
            # Location where the iteration number starts and ends
            start_ind = file.find('-') + 1
            end_ind = file.find('.')
            if int(file[start_ind:end_ind]) not in ignore:
                os.remove(specific_model_dir + '/' + file)
    return ignore_dict


def save_to_dict(models, dir, specific_model_dir, file_name):
    # save the dictionary to pkl file to quick load for next time
    with open(os.path.join(specific_model_dir, file_name + '.pkl'),
              'wb') as f:
        if file_name in ['test_dict', 'gt_listener_test_dict',
                         'gt_listener_val_dict']:
            cPickle.dump(models, f)
        else:
            cPickle.dump(models[dir], f)


def load_dict(specific_model_dir, file_name):
    with open(os.path.join(specific_model_dir, file_name + '.pkl'), 'rb') as f:
        model = cPickle.load(f)
    return model


def read_models_to_plot():
    black_list = ['R0.5_CAP0_V0.1_BSL_gt_C.5_LR5e-3_D0.9_E15_BS128_SE1',
                  'R0.7_CAP0_V0.1_BSL_gt_C.3_LR5e-3_D0.8_E50_SE0_BS128',
                  'MS0.0004_T1_P0.5_C.9996_LR5e-3_D0.75_E15_SE0_SR0_SL0_BS128',
                  'MS0.00033_T1_P0.5_C.99967_LR5e-3_D0.75_E15_SE0_SR0_SL0_BS128',
                  'MS0.00025_T1_P0.5_C.99975_LR5e-3_D0.75_E15_SE0_SR0_SL0_BS128',
                  'G0.0004_T1_C.9996_LR5e-3_D0.75_E15_SE0_SR0_SL0_BS128']
    model_dirs = []
    dirs = list()
    opt = opts()
    for key in vars(opt):
        assert (key.startswith('model')), "Error message"
        # Extract history file name. List of dirs from the same kind
        # of model (e.g. same gumbel temp with different cider weight)
        model_dir = (getattr(opt, key))  # get dir path
        for dir in os.listdir(model_dir):
            print('Start working on [{}]'.format(dir))
            if dir in black_list: continue
            if 'BS50' in dir: continue
            if 'E50' in dir: continue
            dirs.append(dir)
            model_dirs.append(model_dir)
    return model_dirs, dirs



def model_doesnt_need_update(val_dict_name, specific_model_dir,
                             force_new_dict, num_models,
                             min_models_in_dir):
    if (val_dict_name + '.pkl') not in os.listdir(specific_model_dir):
        return False
    if (num_models <= min_models_in_dir and not force_new_dict
            and (val_dict_name + '.pkl') in os.listdir(specific_model_dir)):
        return True
    return False


def create_json(opt, handle_specific_model=0):

    if handle_specific_model:
        model_dir, dir = handle_specific_model.rsplit("/", 1)
        model_dirs = [model_dir]
        dirs = [dir]
    else:
        model_dirs, dirs = read_models_to_plot()
    if dirs[0] == 'log_fc_con':
        dir = dirs[0]
        specific_model_dir = handle_specific_model
        with open(handle_specific_model + '/' + 'histories_fc_con.pkl',
                  'rb') as f:
            histories = cPickle.load(f)
        with open(specific_model_dir + '/' + 'infos_fc_con.pkl', 'rb') as f:
            infos = cPickle.load(f)

        collect_results = {}
        splits = ['val', 'test']
        for iteration in histories['val_result_history']:
            for split in splits:
                recall_split = 't2i_r10_' + split
                cider_split = 'cider_' + split
                bleu_split = 'bleu4_' + split
                iteration_split = 'iteration_' + split
                if not collect_results.get('bleu4_test'):
                    collect_results[iteration_split] = [iteration]
                    collect_results[recall_split] = [histories['val_result_history'][iteration]['loss'][split]['t2i_r10']]
                    collect_results[cider_split] = [histories['val_result_history'][iteration]['lang_stats'][split]['CIDEr']]
                    collect_results[bleu_split] = [histories['val_result_history'][iteration]['lang_stats'][split]['Bleu_4']]
                    if split == 'test':
                        collect_results[recall_split] = [histories['val_result_history'][iteration]['loss'][split]['t2i_r1']]
                        collect_results[recall_split] = [histories['val_result_history'][iteration]['loss'][split]['t2i_r5']]

                else:
                    collect_results[iteration_split].append(iteration)
                    collect_results[recall_split].append(histories['val_result_history'][iteration]['loss'][split]['t2i_r10'])
                    collect_results[cider_split].append(histories['val_result_history'][iteration]['lang_stats'][split]['CIDEr'])
                    collect_results[bleu_split].append(histories['val_result_history'][iteration]['lang_stats'][split]['Bleu_4'])
                    if split == 'test':
                        collect_results[recall_split].append(histories['val_result_history'][iteration]['loss'][split]['t2i_r10'])
                        collect_results[recall_split].append(histories['val_result_history'][iteration]['loss'][split]['t2i_r10'])

        # Find best recall @ 10 in validation set
        argmax_of_val_recall = np.argmax(collect_results['t2i_r10_val'])
        argmax_of_cider_recall = np.argmax(collect_results['cider_val'])
        argmax_of_bleu4_recall = np.argmax(collect_results['bleu4_val'])
        args = [argmax_of_val_recall, argmax_of_cider_recall, argmax_of_bleu4_recall]
        # Create Json
        json_dictionary = {}
        model_best_by = ['t2i_r10', 'cider', 'bleu4']
        for i, best_by in enumerate(model_best_by):
            for split in splits:
                for metric in model_best_by:
                    metric_and_split = metric + '_' + split
                    if not json_dictionary.get(best_by):
                        json_dictionary[best_by] = {metric_and_split: collect_results[metric_and_split][args[i]]}
                    else:
                        json_dictionary[best_by].update({
                            metric_and_split: collect_results[metric_and_split][
                                args[i]]})

        # opts
        for arg in vars(infos['opt']):
            if not json_dictionary.get('opt'):
                json_dictionary['opt'] = {
                    arg: getattr(infos['opt'], arg)}
            else:
                json_dictionary['opt'].update(
                    {arg: getattr(infos['opt'], arg)})
        # save dict in json_dir
        if opt:
            json_dir = os.path.join(os.path.split(
                os.path.split(opt.checkpoint_path)[0])[0], 'json_dir')
        else:
            json_dir = '/cortex/users/gilad/DiscCaptioning_files/' \
                       'our_trained_models/models/json_dir'

        # full path
        json_dictionary['full_path'] = specific_model_dir
        with open(os.path.join(json_dir, dir + '.json'), 'w') as f:
            json.dump(json_dictionary, f)
        print("json file was created for {}".format(
            specific_model_dir))



    else:
        for index, model_dir in enumerate(model_dirs):
            dir = dirs[index]
            specific_model_dir = model_dir + '/' + dir
            if not (os.path.isfile((os.path.join(
                    specific_model_dir + '/' + 'test_dict.pkl')))):
                print('skipped {}'.format(specific_model_dir))
                continue
            with open(specific_model_dir + '/' + 'val_dict.pkl', 'rb') as f:
                val_dict = cPickle.load(f)
            with open(specific_model_dir + '/' + 'test_dict.pkl', 'rb') as f:
                test_dict = cPickle.load(f)
            try:
                listener_weight = re.findall("\d+\.\d+", dir)[0]
                with open(specific_model_dir + '/' + 'infos_att_d' +
                          listener_weight + '.pkl', 'rb') as f:
                    infos = cPickle.load(f)
            except:
                listener_weight = re.findall("\d", dir)[0]
                with open(specific_model_dir + '/' + 'infos_att_d' +
                          listener_weight + '.pkl', 'rb') as f:
                    infos = cPickle.load(f)

            for model in test_dict:
                model_name = model
            splits = ['val', 'test']
            model_best_by = ['t2i_r10', 'cider', 'bleu4']
            metrics = ['t2i_r10', 'cider_score', 'bleu4']
            json_dictionary = {}
            for split in splits:
                if split == 'val':
                    for best_model in model_best_by:
                        for i, metric in enumerate(metrics):
                            metric_and_split = model_best_by[i] + '_' + split
                            if not json_dictionary.get(best_model):
                                json_dictionary[best_model] = {
                                    metric_and_split: val_dict[metric][
                                        val_dict['best_iteration_by'][
                                            best_model]]}
                            else:
                                json_dictionary[best_model].update({
                                    metric_and_split: val_dict[metric][
                                        val_dict['best_iteration_by'][
                                            best_model]]})

                elif split == 'test':
                    for best_model in model_best_by:
                        metric_and_split = ['t2i_r10_test', 't2i_r5_test',
                                            't2i_r1_test', 'cider_test',
                                            'bleu4_test']
                        json_dictionary[best_model].update({
                            metric_and_split[0]:test_dict[model_name][
                                best_model]['loss']['t2i_r10']})
                        json_dictionary[best_model].update({
                            metric_and_split[1]: test_dict[model_name][
                                best_model]['loss']['t2i_r5']})
                        json_dictionary[best_model].update({
                            metric_and_split[2]: test_dict[model_name][
                                best_model]['loss']['t2i_r1']})
                        json_dictionary[best_model].update({
                            metric_and_split[3]:test_dict[model_name][
                                best_model]['lang_stats']['CIDEr']})
                        json_dictionary[best_model].update({
                            metric_and_split[4]:test_dict[model_name][
                                best_model]['lang_stats']['Bleu_4']})


            # opts
            for arg in vars(infos['opt']):
                if not json_dictionary.get('opt'):
                    json_dictionary['opt'] = {arg: getattr(infos['opt'], arg)}
                else:
                    json_dictionary['opt'].update({arg: getattr(infos['opt'], arg)})
            # save dict in json_dir
            if opt:
                json_dir = os.path.join(os.path.split(os.path.split(
                    opt.checkpoint_path)[0])[0], 'json_dir')
            else:
                json_dir = '/cortex/users/gilad/DiscCaptioning_files/' \
                           'our_trained_models/models/json_dir'

            # full path
            json_dictionary['full_path'] = specific_model_dir
            with open(os.path.join(json_dir, dir + '.json'), 'w') as f:
                json.dump(json_dictionary, f)
            print("json file was created for {}" .format(dir))

def create_model_metrics(specific_model_dir, dir, models):
    # Search for history file name in the model dir
    num_matching_history_files = 0
    for file in os.listdir(specific_model_dir):
        if file.startswith("histories_att_d"):
            num_matching_history_files += 1
            history_file = file
            assert (num_matching_history_files <= 1)
    with open(os.path.join(specific_model_dir + '/'
                           + history_file), 'rb') as history_f:
        history = cPickle.load(history_f)

    ckpt = 1
    for i in history['val_result_history'].keys():
        if i % ckpt == 0:
            cider_score = [history['val_result_history'][i] \
                               ['lang_stats']['CIDEr']]
            bleu4 = [history['val_result_history'][i] \
                         ['lang_stats']['Bleu_4']]
            t2i_r10 = [history['val_result_history'][i] \
                           ['loss']['t2i_r10']]
            if not models.get(dir):
                models[dir] = {'iteration': [i]}
                models[dir].update({'cider_score': [cider_score]})
                models[dir].update({'bleu4': [bleu4]})
                models[dir].update({'t2i_r10': [t2i_r10]})
            else:
                models[dir]['iteration'].append(i)
                models[dir]['cider_score'].append(cider_score)
                models[dir]['bleu4'].append(bleu4)
                models[dir]['t2i_r10'].append(t2i_r10)

    models[dir].update({'best_iteration_by': {'cider': np.argmax(
        models[dir]['cider_score'])}})
    models[dir]['best_iteration_by'].update({'bleu4': np.argmax(
        models[dir]['bleu4'])})
    models[dir]['best_iteration_by'].update({'t2i_r10': np.argmax(
        models[dir]['t2i_r10'])})

    test_models = remove_unnecessary_models(models, dir, specific_model_dir)
    models[dir].update({'test_models': test_models})

    #  Remove unnecessary models
    save_to_dict(models, dir, specific_model_dir, 'val_dict')
    print("Saved to [%s]" % dir)
    return models

def create_dict(opt, force_new_dict=False, listener=None, split='test',
                only_recall=1, handle_specific_model=None):
    val_dict_name = 'val_dict'
    #  If number of models in directory is greater than this, create new
    #  dictionary
    min_models_in_dir = 5
    # Go through all lind of models (in opt) in each directory there might
    # be some different models.

    # Read set of models for plotting
    if handle_specific_model:
        model_dir, dir = handle_specific_model.rsplit("/", 1)
        model_dirs = [model_dir]
        dirs = [dir]
    else:
        model_dirs, dirs = read_models_to_plot()

    models = {}
    for index, model_dir in enumerate(model_dirs):
        dir = dirs[index]
        specific_model_dir = model_dir + '/' + dir

        # In case that we already created dictionary to plot by its values
        # Also check if the model continued to run after we built the
        # dictionary, if it did built dictionary again
        num_models = len([model_name for model_name in
                          os.listdir(specific_model_dir) if
                          model_name.startswith('alternatingModel-')
                          or model_name.startswith('model-')])
        if model_doesnt_need_update(val_dict_name, specific_model_dir,
                                    force_new_dict, num_models,
                                    min_models_in_dir):
            model_metrics = load_dict(specific_model_dir, val_dict_name)
            models.update({dir: model_metrics})
        else:
            # If it's the first time or we continued to train the model
            models = create_model_metrics(specific_model_dir, dir, models)


        if split == 'test':
            if 'test_dict' not in vars():
                test_dict = eval_test(opt, models, specific_model_dir,
                                      dir, listener, split, num_models,
                                      force_new_dict, only_recall)
            else:
                test_dict.update(eval_test(opt, models, specific_model_dir,
                                           dir, listener, split, num_models,
                                           force_new_dict, only_recall))

        print('\tfinished processing [%s]' % dir)
    return models


def eval_test(opt, models, specific_model_dir, dir, listener, split, num_models,
              force_new_dict=0, only_recall=1):
    sys.path.append('/home/lab/vgilad/PycharmProjects/our_DiscCaption_model')
    min_models_in_dir = 5
    if listener == 'gt':
        if split == 'test':
            file_name = 'gt_listener_test_dict'
        elif split == 'val':
            file_name = 'gt_listener_val_dict'
    else:  # No gt
        file_name = 'test_dict'
    sys.path.append('../coco-caption/pycocoevalcap')
    sys.path.append('../coco-caption')
    annFile = '../coco-caption/annotations/captions_val2014.json'
    test_dict = {}
    iter_to_metric = {}
    if (file_name + '.pkl') in os.listdir(specific_model_dir) \
            and not force_new_dict and num_models <= min_models_in_dir:
        test_dict = load_dict(specific_model_dir, file_name)
        # models.update({dir: model})
    # No test_dict - create it
    else:
        for metric, iteration in models[dir]['test_models'].items():
            #  if only recall is 1 - evaluate only for best recall @ 10 model
            if only_recall and metric in ['cider', 'bleu4']:
                continue
            # We already calculated results for this model
            if iteration in iter_to_metric:
                test_dict[dir].update({metric: test_dict[dir][iter_to_metric
                [iteration]]})
            else:  # Eval results
                max_epoch = 309000
                alternating = 1  # Variable initialization
                for file in os.listdir(specific_model_dir):
                    if 'model-' in file and file[6].isdigit() \
                            and iteration <= max_epoch:
                        alternating = 0
                if not alternating:
                    model_name = '/model-'
                elif alternating:
                    model_name = '/alternatingModel-'
                model_name = specific_model_dir + model_name + \
                             str(iteration) + '.pth'
                for file in os.listdir(specific_model_dir):
                    if file.startswith("infos_att_d") and \
                            file.endswith(str(iteration) + '.pkl'):
                        infos = file
                        start_ind = infos.index('d') + 1
                        end_ind = infos.index('.p')
                        disc_weight = infos[start_ind:end_ind]
                        infos_name = specific_model_dir + '/infos_att_d' + \
                                     disc_weight + '.pkl'
                        break
                if not test_dict.get(dir):
                    test_dict.update({dir: {metric: eval.eval(opt,
                                                              model_name,
                                                              infos_name,
                                                              annFile,
                                                              listener,
                                                              split)}})
                else:
                    test_dict[dir].update({metric: eval.eval(opt, model_name,
                                                             infos_name,
                                                             annFile,
                                                             listener,
                                                             split)})
                iter_to_metric.update({iteration: metric})
        save_to_dict(test_dict, dir, specific_model_dir, file_name)
    return test_dict


def create_dicts_and_json_after_training(opt):
    if isinstance(opt, str):  # Run as stand alone, if not after train
        handle_specific_model = opt
    else:  # Run after train
        handle_specific_model = opt.checkpoint_path
    splits = ['val', 'test']
    for split in splits:
        print("start creating {}_dict.pkl".format(split))
        create_dict(opt, force_new_dict=True, listener=None, split=split,
                    only_recall=0, handle_specific_model=handle_specific_model)
        print("finished creating {}_dict.pkl".format(split))
    print('Start creating json file')
    if isinstance(opt, str):
        create_json(opt=None, handle_specific_model=handle_specific_model)
    else:
        create_json(opt, handle_specific_model=handle_specific_model)
    print('json file has been created \n')


def checkpoint_dir_for_stand_alone():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default='/cortex/users/gilad/DiscCaptioning_files'
                                '/our_trained_models/models/gumbel/G0.0025_T1_C.9975_LR5e-3_D0.75_E15_SE0_new1_BS128',
                        help='Parent directory to load models from')
    opt = parser.parse_args()
    number_models = len(os.listdir(opt.checkpoint_path))
    return opt, number_models


def opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default='/cortex/users/gilad/DiscCaptioning_files/'
                                'our_trained_models/models/multinomial',
                        help='model to luo results from')
    opt = parser.parse_args()
    return opt

def main():

    """
     To create json file for parent directory (bunch of models)
        Change in opts, --model_dir argument, which indicates on the
        location of the parent directory
     To create json file for specific model, enter it for
        handle_specific_model variable
        and to checkpoint_path in checkpoint_dir_for_stand_alone()
     To create dictionaries and json, set 'create_dicts_and_json' = True,
     to create only json file set it to False
     """
    handle_specific_model = \
        '/cortex/users/gilad/DiscCaptioning_files/our_trained_models/models' \
        '/gumbel/G0.0025_T1_C.9975_LR5e-3_D0.75_E15_SE0_new1_BS128'
    create_dicts_and_json = True
    opt, number_models = checkpoint_dir_for_stand_alone()
    if create_dicts_and_json:
        if number_models <= 1 or handle_specific_model:  # Single model
            create_dicts_and_json_after_training(opt)
        else:  # Multiple models
            for model_dir in os.listdir(opt.checkpoint_path):
                current_model = os.path.join(opt.checkpoint_path, model_dir)
                create_dicts_and_json_after_training(current_model)
    else:  # Create only json
        if number_models <= 1 or handle_specific_model:  # Single model
            create_json(opt, handle_specific_model=handle_specific_model)
        else:  # Multiple models
            for model_dir in os.listdir(opt.checkpoint_path):
                current_model = os.path.join(opt.checkpoint_path, model_dir)
                create_json(handle_specific_model=current_model)


if __name__ == '__main__':
    main()
