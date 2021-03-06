import hydra
from src.utils.composeTransform import getTransform


def train(cfg):
    # Init torch transforms for datamodule
    transform = getTransform(cfg)

    # Init Lightning datamodule
    dataModule = hydra.utils.instantiate(cfg['dataModule'], transform=transform, _recursive_=False)

    # # Init Lightning model
    module = hydra.utils.instantiate(cfg['module'])

    # Init Lightning loggers
    if "logger" in cfg:
        logger = []
        for _, logger_cfg in cfg['logger'].items():
            logger.append(hydra.utils.instantiate(logger_cfg))
    
    # Init Lightning callbacks
    if "callbacks" in cfg:
        callbacks = []
        for _, callbacks_cfg in cfg['callbacks'].items():
            callbacks.append(hydra.utils.instantiate(callbacks_cfg))

    # Init Lightning trainer
    trainer = hydra.utils.instantiate(cfg['trainer'], logger=logger, callbacks=callbacks)
    trainer.fit(module, dataModule)