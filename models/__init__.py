from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .FCModel import FCModel
from .AttModel import *
from .VSEFCModel import VSEFCModel


__all__ = ['setup', 'load', 'AlternatingJointModel']

# def setup(opt, model_name, caption = True):
def setup(opt, model_name, model_type = 'caption_model'):

    # if caption:
    if model_type == 'caption_model':
        if model_name == 'fc':
            model = FCModel(opt)
        # Att2in model with two-layer MLP img embedding and word embedding
        elif model_name == 'att2in2':
            model = Att2in2Model(opt)
        else:
            raise Exception(
                "Caption model not supported: {}".format(model_name))
    # else:
    elif model_type == 'vse_model':
        if model_name == 'fc':
            model = VSEFCModel(opt)
        else:
            raise Exception("VSE model not supported: {}".format(model_name))

    return model

def load(model, opt, iteration=None):
    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        if torch.cuda.is_available():
            if iteration:
                utils.load_state_dict(model, torch.load(os.path.join(
                    opt.start_from, 'model-' + iteration + '.pth')))
            else:
                utils.load_state_dict(model, torch.load(os.path.join(
                    opt.start_from, 'model.pth')))
        else:
            utils.load_state_dict(model, torch.load(
                os.path.join(opt.start_from, 'model.pth'), map_location='cpu'))

from .AlternatingJointModel import *