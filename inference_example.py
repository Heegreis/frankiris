from src.inference import inference_module

config_file = "outputs/2021-06-13/01-10-54/.hydra/config.yaml"
ckpt = 'ff.ckpt'
inference_module(config_file)

