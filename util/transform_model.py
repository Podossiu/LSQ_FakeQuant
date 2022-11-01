import torch
import copy
from quan import *
from model import *
@torch.no_grad()
def transform_model(model, args):
    model_clone = prepare_qat_model(args.arch, pre_trained = False, mode = args.mode, distillation = False)
    model_clone.load_state_dict(model.state_dict())
    '''
    for n,m in model.named_modules():
        if hasattr(m, "soft_mask"):
            m.soft_mask = None

    model_clone = copy.deepcopy(model)
    '''
    '''
    if args.mode == "slsq":
        for n, m in model_clone.named_modules():
            if hasattr(m, "weight_fake_quant"):
                m.weight.data = m.weight_fake_quant(m.weight)
    '''
    return model_clone

