import hydra


def train(cfg):
    # Init torch transforms for datamodule
    train_transforms = []
    for transform_cfg in cfg.transforms.train_transforms:
        train_transforms.append(hydra.utils.instantiate(transform_cfg))
    val_transforms = []
    for transform_cfg in cfg.transforms.val_transforms:
        val_transforms.append(hydra.utils.instantiate(transform_cfg))
    test_transforms = []
    for transform_cfg in cfg.transforms.test_transforms:
        test_transforms.append(hydra.utils.instantiate(transform_cfg))

    # Init Lightning datamodule
    dataModule = hydra.utils.instantiate(cfg.dataModule, train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms)

    # Init Lightning model
    module = hydra.utils.instantiate(cfg.module)

    # Init Lightning loggers
    if "logger" in cfg:
        logger = []
        for _, logger_cfg in cfg.logger.items():
            logger.append(hydra.utils.instantiate(logger_cfg))
    
    # Init Lightning callbacks
    if "callbacks" in cfg:
        callbacks = []
        for _, callbacks_cfg in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callbacks_cfg))

    # Init Lightning trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(module, dataModule)