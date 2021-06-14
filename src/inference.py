from pathlib import Path

import hydra
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from src.utils.composeTransform import getTransform


class inference_module:
    def __init__(self, config_file, ckpt):
        config_file = Path(config_file)
        config_path = '../' + config_file.parent.as_posix()
        config_name = config_file.stem
        # initialize the Hydra subsystem.
        # This is needed for apps that cannot have a standard @hydra.main() entry point
        initialize(config_path=config_path)
        cfg = compose(config_name, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            del cfg["hydra"]
        # print(OmegaConf.to_yaml(cfg))

        # Init torch transforms for datamodule
        transform = getTransform(cfg)

        # Init Lightning datamodule
        dataModule = hydra.utils.instantiate(
            cfg['dataModule'], transform=transform, _recursive_=False
        )

        ## Init Lightning model
        self.module = hydra.utils.instantiate(
            cfg['module'], dataModule=dataModule, _recursive_=False
        )
        checkpoint = torch.load(f=ckpt)
        self.module.load_state_dict(checkpoint['state_dict'])
        self.module.eval()

    def predict(self, x):

        y_hat = self.module()

        # 要建立class label對照表，預計在train的時候就寫入在cfg或額外的class.csv

        return y_hat
