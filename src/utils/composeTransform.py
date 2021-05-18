import hydra
from src.utils.cfgfix import fix_tuple
from torchvision import transforms
import torch.nn as nn


def getTransform(cfg):
    stages = ['train', 'val', 'test']
    transform = {}
    for stage in stages:
        transform[stage] = {}
        audio_transform_list = []
        vision_transform_list = []
        for transform_cfg in cfg.transforms[stage]:
            fix_tuple(transform_cfg)
            if transform_cfg['_target_'].split('.')[0] == 'torchaudio':
                audio_transform_list.append(hydra.utils.instantiate(transform_cfg))
            elif transform_cfg['_target_'].split('.')[0] == 'torchvision':
                vision_transform_list.append(hydra.utils.instantiate(transform_cfg))

        transform[stage]['audio'] = nn.Sequential(*audio_transform_list)
        transform[stage]['vision'] = transforms.Compose(vision_transform_list)

    return transform