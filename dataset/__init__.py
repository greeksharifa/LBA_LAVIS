import importlib

from pathlib import Path

from config.configs import Config
from util.logger import get_logger
from util.utils import data_print

def load_dataset(cfg: Config):#, output_dir: Path):
    # load dataset class
    dataset_name = cfg.dataset_cfg.dataset_name
    module_name = f"dataset.{dataset_name}"
    dataset_class = load_class(module_name, dataset_name)
    
    logger = get_logger()
    logger.info("="*100)
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Dataset class: {dataset_class}")

    # create dataset instance
    dataset = dataset_class(cfg)#, output_dir=output_dir)
    sample = dataset[0]
    logger.info(f"Data sample:")

    for k, v in sample.items():
        logger.info(f"{k:15s}: {data_print(v)}")
        
    logger.info("="*100)

    return dataset



def load_class(module_name, class_name):
    module = importlib.import_module(module_name)  # Import module dynamically
    return getattr(module, class_name)  # Get class reference



if __name__ == "__main__":
    # Usage
    module_name = "dataset.MMLU"  # Replace with your module
    class_name = "MMLU"  # Replace with your class

    MMLU = load_class(module_name, class_name)

    # Create an instance
    instance = MMLU()
