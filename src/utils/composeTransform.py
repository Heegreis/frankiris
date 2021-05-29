import hydra
from src.utils.cfgfix import fix_tuple
import torchvision
import torchaudio
from torchvision import transforms
import torch.nn as nn


def instanceTransform(transform_cfg):
    target = ''
    p = ''
    for key, value in transform_cfg.items():
        if key == '_target_':
          target = value
        else:
            p = p + key + '=' + str(value) + ','
    p = p[:-1]
    cmd = eval(target + '(' + p + ')')
    return cmd

def getTransform(cfg):
    # for every stage and transform type, if empty just set None
    stages = ['train', 'val', 'test']
    transform = {}
    for stage in stages:
        transform[stage] = {}
        transform[stage]['audio'] = None
        transform[stage]['vision'] = None
        if "transforms" in cfg:
            audio_transform_list = []
            vision_transform_list = []
            if stage in cfg['transforms']:
                for transform_cfg in cfg['transforms'][stage]:
                    # fix_tuple(transform_cfg)
                    if transform_cfg['_target_'].split('.')[0] == 'torchaudio':
                        audio_transform_list.append(instanceTransform(transform_cfg))
                    elif transform_cfg['_target_'].split('.')[0] == 'torchvision':
                        vision_transform_list.append(instanceTransform(transform_cfg))
            if len(audio_transform_list) > 0:
                transform[stage]['audio'] = nn.Sequential(*audio_transform_list)
                
            if len(vision_transform_list) > 0:
                transform[stage]['vision'] = transforms.Compose(vision_transform_list)

    return transform