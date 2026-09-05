import os
import yaml
from Chess import logger
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path


@ensure_annotations
def read_yaml(path: Path) ->ConfigBox :
    try :
        with open(path) as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Config loaded from {path}")
            return ConfigBox(config)
    except BoxValueError :
        raise ValueError("yaml file is empty ")
    except Exception as e:
        raise e


@ensure_annotations
def  create_directories (path: list , verbose: bool = True )  :
    for p in path :
        os.makedirs(p, exist_ok=True)
        if verbose:
            logger.info(f"Created directory in {p}")
