import hydra


def train(cfg):
    # Init Lightning datamodule
    dataModule = hydra.utils.instantiate(cfg.dataModule)

    # Init Lightning model
    module = hydra.utils.instantiate(cfg.module)

    # Init Lightning loggers
    if "logger" in cfg:
        logger = []
        for _, loggerCfg in cfg.logger.items():
            logger.append(hydra.utils.instantiate(loggerCfg))
            print(logger[0])
    
    # init trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.fit(module, dataModule)