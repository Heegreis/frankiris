import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg : DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train

    # print(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    train(cfg)
    


if __name__ == "__main__":
    main()