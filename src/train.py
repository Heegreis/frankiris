import hydra
from src.utils.composeTransform import getTransform

import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
import time
import csv


def train(cfg):
    # Init torch transforms for datamodule
    transform = getTransform(cfg)

    # Init Lightning datamodule
    dataModule = hydra.utils.instantiate(cfg['dataModule'], transform=transform, _recursive_=False)

    # # Init Lightning model
    module = hydra.utils.instantiate(cfg['module'], dataModule=dataModule, _recursive_=False)

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
    else:
        callbacks = None

    if "mode" in cfg:
        if cfg['mode'] == 'inference':
            # inference
            # hydra.utils.instantiate(cfg['module'], dataModule=dataModule, _recursive_=False)
            # model = LightningModule.load_from_checkpoint(hydra.utils.get_original_cwd() + '/ff.ckpt', torch_module=cfg['module']['torch_module'], dataModule=dataModule)
            checkpoint = torch.load(f=hydra.utils.get_original_cwd() + '/ff.ckpt')
            module.load_state_dict(checkpoint['state_dict'])

            module.to('cuda:0')

            module.eval()

            with open(hydra.utils.get_original_cwd() + '/submit.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Filename','Barking','Howling','Crying','COSmoke','GlassBreaking','Other'])

                # for loop data
                root = Path(hydra.utils.get_original_cwd() + '/dataset/public_test')
                tt = 0
                stime = time.time()
                for file in sorted(root.iterdir()):
                    waveform, sample_rate = torchaudio.load(file)
                    log_mel_spec = (dataModule.train.to_mel_spectrogram(waveform) + torch.finfo(torch.float).eps).log()
                    x = torch.Tensor(log_mel_spec).to('cuda:0')
                    # x = torch.Tensor(log_mel_spec)

                    y_hat = module(x)
                    y_hat = F.softmax(y_hat, dim=1)

                    # y_hat = torch.argmax(y_hat, dim=1)
                    print(y_hat)
                    tt = tt + 1
                    ans = [file.stem]
                    y_hat = y_hat[0].tolist()
                    sub_y = [y_hat[0], y_hat[4], y_hat[2], y_hat[1], y_hat[3], y_hat[5]]
                    ans.extend(sub_y)
                    print(ans)
                    writer.writerow(ans)
            ttime = time.time() - stime
            atime = ttime / tt
            print(tt)
            print(f'totle time: {ttime}')
            print(f'average time: {atime}')
        elif cfg['mode'] == 'test':
            checkpoint = torch.load(f=hydra.utils.get_original_cwd() + '/ff.ckpt')
            module.load_state_dict(checkpoint['state_dict'])
            module.to('cuda:0')
            module.eval()
            root = Path(hydra.utils.get_original_cwd() + '/dataset/Tomofun_dog_sound_20210513_random_seed_0/train')
            print(dataModule.train.class_to_idx)
            # for class_dir in root.iterdir():
            #     class_name = class_dir.name
            #     for file in class_dir.iterdir():
            #         waveform, sample_rate = torchaudio.load(file)
            #         log_mel_spec = (dataModule.train.to_mel_spectrogram(waveform) + torch.finfo(torch.float).eps).log()
            #         x = torch.Tensor(log_mel_spec).to('cuda:0')
            #         # x = torch.Tensor(log_mel_spec)

            #         y_hat = module(x)
            #         # y_hat = F.softmax(y_hat, dim=1)

            #         y_hat = torch.argmax(y_hat, dim=1)
            #         dataModule.train
            #         print(class_name)
        else:
            # Init Lightning trainer
            trainer = hydra.utils.instantiate(cfg['trainer'], logger=logger, callbacks=callbacks)
            trainer.fit(module, dataModule)

    