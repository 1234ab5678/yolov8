#from mmdet.models.builder import BACKBONES as det_BACKBONES
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parent_parentdir)
from .fasternet import FasterNet
from torch.hub import load_state_dict_from_url

model_urls = {
    'fasternet_s': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_s-epoch.299-val_acc1.81.2840.pth',
    'fasternet_m': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_m-epoch.291-val_acc1.82.9620.pth',
    'fasternet_l': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_l-epoch.299-val_acc1.83.5060.pth',
}

#@det_BACKBONES.register_module()
def fasternet_s(**kwargs):
    model = FasterNet(
        mlp_ratio=2.0,
        embed_dim=128,
        depths=(1, 2, 13, 2),
        drop_path_rate=0.15,
        act_layer='RELU',
        fork_feat=True,
        **kwargs
        )

    return model


#@det_BACKBONES.register_module()
def fasternet_m(**kwargs):
    model = FasterNet(
        mlp_ratio=2.0,
        embed_dim=144,
        depths=(3, 4, 18, 3),
        drop_path_rate=0.2,
        act_layer='RELU',
        fork_feat=True,
        **kwargs
        )

    return model


#@det_BACKBONES.register_module()
def fasternet_l(**kwargs):
    model = FasterNet(
        mlp_ratio=2.0,
        embed_dim=192,
        depths=(3, 4, 18, 3),
        drop_path_rate=0.3,
        act_layer='RELU',
        fork_feat=True,
        **kwargs
        )

    return model
