import os
import yaml
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
import logging
import json
logger = logging.getLogger("ChessLogger")
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
@ensure_annotations
def save_json (path : Path , data : dict) :

    with open(path, 'w') as f:
        json.dump(data, f , indent=4)
        logger.info(f"json file  saved in  : {path}")