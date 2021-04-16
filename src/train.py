import hydra


def train(cfg):
    # Init Lightning datamodule
    dataModule = hydra.utils.instantiate(cfg.dataModule)

    # Init Lightning model
    module = hydra.utils.instantiate(cfg.module)
    
    # init trainer
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(module, dataModule)