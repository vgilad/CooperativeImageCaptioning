
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataloader import *
import sys
import torch.optim as optim
from misc import utils


def load_optimizer_path(opt, curr_turn=None):
    if opt.is_alternating:
        path_exist = os.path.isfile(os.path.join(
            opt.start_from, curr_turn + '_optimizer.pth'))
        if path_exist:
            return os.path.join(opt.start_from, curr_turn + '_optimizer.pth')
        else:
            return None
    else:  #
        if opt.start_from is not None:
            return os.path.join(opt.start_from, 'optimizer.pth')
        else:
            return None


def define_optimizer(model, opt):
    return optim.Adam([p for p in model.parameters()], lr=opt.learning_rate,
                      weight_decay=opt.weight_decay)


def load_state_dict(optimizer, optimizer_path, agent=''):
    if torch.cuda.is_available():
        optimizer_state_dic = torch.load(optimizer_path)
    else:  # CPU()
        optimizer_state_dic = torch.load(optimizer_path,
                                             map_location=
                                             lambda storage, loc:
                                             storage)
    optimizer.load_state_dict(optimizer_state_dic)
    print(f'\n Loaded {agent} optimizer from {optimizer_path} \n')
    return optimizer


def load_optimizer_from_checkpoint(opt, curr_turn, optimizer):
    optimizer_path = load_optimizer_path(opt, curr_turn)
    optimizer = load_state_dict(optimizer, optimizer_path, curr_turn)
    return optimizer


def define_speaker_optimizer_joint_training(
        model, opt, start_from_exist, optimizer_dict, curr_turn):
    optimizer = define_optimizer(model.caption_generator, opt)
    # If we need to start from previous snapshot
    if start_from_exist:
        if load_optimizer_path(opt, curr_turn):
            # If we have a previous version of 'speaker' optimizer
            optimizer = load_optimizer_from_checkpoint(opt, curr_turn,
                                                       optimizer)
        # Use the optimizer at the end of the second phase in case
        # we don't use shared embedding, else use new optimizer
        elif not opt.share_embed:
            optimizer_path = opt.speaker_stage_2_optimizer_path
            optimizer = load_state_dict(
                optimizer, optimizer_path, curr_turn)
    else:
        print('Loaded new "speaker" optimizer')
    optimizer_dict[curr_turn] = optimizer
    return optimizer_dict


def define_listener_optimizer_joint_training(
        model, opt, start_from_exist, optimizer_dict, curr_turn):
    optimizer = define_optimizer(model.vse, opt)
    # If we need to start from previous snapshot
    if start_from_exist:
        if load_optimizer_path(opt, curr_turn):
            # if we have a previous version of 'listener' optimizer
            optimizer = load_optimizer_from_checkpoint(opt, curr_turn,
                                                       optimizer)
        # load optimizer from the end of phase 1
        elif not opt.share_embed:
            optimizer_path = os.path.join(
                os.path.split(opt.initialize_retrieval)[0], 'optimizer.pth')
            # Copied from listener training phase
            optimizer = load_state_dict(
                optimizer, optimizer_path, curr_turn)
        else:
            print('\n Using new "listener" optimizer \n')
        if opt.retrieval_reward == 'reinforce':
            optimizer_dict[curr_turn] = optimizer
        else:  # Any other method besides reinforce
            optimizer_dict['speaker'] = {
                'speaker': optimizer_dict['speaker'],
                'listener': optimizer}
            # In speaker turn we will use both optimizers
            opt.alternating_turn.remove('listener')
    return optimizer_dict


def define_pretraining_listener_optimizer(model, opt, start_from_exist,
                                          optimizer_dict, optimizer_exist):
    # First phase (training the listener alone)
    optimizer = define_optimizer(model.vse, opt)
    if start_from_exist and optimizer_exist:
        optimizer_path = os.path.join(
            opt.start_from, 'optimizer.pth')
        optimizer = load_state_dict(optimizer, optimizer_path)
    else:
        print('Optimizer param group number not matched? '
              'There must be new parameters. Reinit the optimizer.')
    optimizer_dict['optimizer'] = optimizer
    return optimizer_dict


def define_pretraining_speaker_optimizer(model, opt, start_from_exist,
                                         optimizer_dict, optimizer_exist):
    # Second phase MLE phase
    optimizer = define_optimizer(model.caption_generator, opt)
    if start_from_exist and optimizer_exist:
        optimizer_path = os.path.join(
            opt.start_from, 'optimizer.pth')
        optimizer = load_state_dict(optimizer, optimizer_path)
    else:
        print('Load new optimizer.')
    optimizer_dict['optimizer'] = optimizer
    return optimizer_dict


def define_only_speaker_optimizer(model, opt, start_from_exist, 
                                  optimizer_dict, optimizer_exist):
    # Third phase finetune, only speaker training without listener training
    new_optimizer = define_optimizer(model.caption_generator, opt)
    if start_from_exist:  # if we need to start from previous snapshot
        if optimizer_exist:
            # if we have a previous version of 'speaker' optimizer
            old_optimizer_path = os.path.join(
                opt.start_from, 'optimizer.pth')
            new_optimizer = load_state_dict(
                new_optimizer, old_optimizer_path, "speaker")
        # Use the optimizer from the end of second phase
        elif not opt.share_embed:
            old_optimizer_path = opt.speaker_stage_2_optimizer_path
            new_optimizer = load_state_dict(
                new_optimizer, old_optimizer_path, "speaker")
    optimizer_dict['optimizer'] = new_optimizer
    return optimizer_dict



def load_optimizer(model, opt):
    # Check for all cases if we start from previous checkpoint
    start_from_exist = (vars(opt).get('start_from', None) is not None)
    optimizer_dict = {}
    if opt.is_alternating:
        for curr_turn in opt.alternating_turn:
            if curr_turn == 'speaker':
                optimizer_dict = define_speaker_optimizer_joint_training(
                    model, opt, start_from_exist, optimizer_dict, curr_turn)
            elif curr_turn == 'listener':
                optimizer_dict = define_listener_optimizer_joint_training(
                    model, opt, start_from_exist, optimizer_dict, curr_turn)

    elif not opt.is_alternating:
        # Phase 1 - Listener training on ground truth.
        # Phase 2 - Speaker training with MLE.
        # Phase 3 - Reinforce training with frozen Listener and Cider
        # optimization.

        # Check if we continue from previous optimizer
        optimizer_exist = load_optimizer_path(opt)

        if opt.phase == 1:  # First phase (training the listener alone)
            optimizer_dict = define_pretraining_listener_optimizer(
                model, opt, start_from_exist, optimizer_dict, optimizer_exist)

        elif opt.phase == 2:  # Second phase MLE phase
            optimizer_dict = define_pretraining_speaker_optimizer(
                model, opt, start_from_exist, optimizer_dict, optimizer_exist)

        # Third phase finetune, only speaker training without listener training
        elif opt.phase == 3:
            optimizer_dict = define_only_speaker_optimizer(
                model, opt, start_from_exist, optimizer_dict, optimizer_exist)
            
    else:  # No such phase
        assert opt.phase in [1, 2, 3], f'phase has to be 1,2 or 3 ' \
                                       f'but got {opt.phase}'

    return optimizer_dict


def save_optimizer(opt, optimizer_dict):

    if opt.is_alternating:  # If alternating save all optimizers
        if opt.retrieval_reward == 'reinforce':
            for agent in optimizer_dict.keys():  # for every optimizer / agent
                optimizer_path = os.path.join(opt.checkpoint_path,
                                              agent + '_optimizer.pth')
                torch.save(optimizer_dict[agent].state_dict(),
                           optimizer_path)
                print(f"\n Optimizer {agent} saved to"
                      f" {optimizer_path} \n")
        else:  # gumbel and multinomial case
            for agent in optimizer_dict.keys():
                # Two optimizers (speaker and listener)
                # in gumbel optimization
                if agent == 'speaker':
                    for agentIn in optimizer_dict['speaker'].keys():
                        optimizer_path = os.path.join(
                            opt.checkpoint_path,
                            agentIn + '_optimizer.pth')
                        torch.save(optimizer_dict['speaker'][
                                       agentIn].state_dict(),
                                   optimizer_path)
                        print(f"\n Optimizer {agentIn} "
                              f"saved to {optimizer_path} \n")

    else:  # Non alternating
        # In non-alternating case there is only one optimizer
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
        torch.save(optimizer_dict['optimizer'].state_dict(), optimizer_path)
        print(f'\n optimizer saved to {optimizer_path}')


def zeroing_optimizer(opt, optimizer_dict, optimizer):
    # If we use gumbel or multinomial we update both optimizers
    if opt.retrieval_reward != 'reinforce' and opt.is_alternating:
        for agent in optimizer_dict['speaker'].keys():
            optimizer_dict['speaker'][agent].zero_grad()
    else:  # Reinforce
        optimizer.zero_grad()


def update_optimizer(optimizer_dict, optimizer, opt):
    # If we use gumbel or multinomial we update both optimizers
    if opt.retrieval_reward != 'reinforce' and opt.is_alternating:
        for agent in optimizer_dict['speaker'].keys():
            utils.clip_gradient(optimizer_dict['speaker'][agent],
                                opt.grad_clip)
            optimizer_dict['speaker'][agent].step()
    else:  # sheriff or listener turn
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
