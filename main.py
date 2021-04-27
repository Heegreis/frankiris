import hydra
from omegaconf import DictConfig, OmegaConf
from clearml import Task


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg : DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train

    task = Task.init(project_name="examples", task_name="pytorch lightning example")
    logger = task.get_logger()
    logger.report_text("You can view your full hydra configuration under Configuration tab in the UI")

    # print(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    train(cfg)
    


if __name__ == "__main__":
    main()