import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(num_channels, prefer=32):
    g = min(prefer, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return g

def replace_bn3d_with_gn(module: nn.Module, prefer_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            c = child.num_features
            g = _pick_groups(c, prefer_groups)
            gn = nn.GroupNorm(
                num_groups=g,
                num_channels=c,
                eps=child.eps,
                affine=True,
            )
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            replace_bn3d_with_gn(child, prefer_groups)

            
def replace_first(root: nn.Module, pred, new_module: nn.Module) -> bool:
    for name, child in root.named_children():
        if pred(child):
            setattr(root, name, new_module)
            return True
        if replace_first(child, pred, new_module):
            return True
    return False
