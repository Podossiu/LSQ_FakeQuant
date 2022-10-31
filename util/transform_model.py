import torch
import copy
from quan import *

@torch.no_grad()
def transform_model(model, args):
    transformed_model = copy.deepcopy(model)
    if args.mode == "qil":
        for n, m in transformed_model.named_children():
            for n1, m1 in m.named_children():
                for n2, m2 in m1.named_children():
                    if hasattr(m2, "weight_fake_quant"):
                        m2.weight.data.copy_(m2.weight_fake_quant(m2.weight).data)
    return transformed_model

