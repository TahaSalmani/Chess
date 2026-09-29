from Chess import logger
from Chess.components.torch_base_model import TransformerEncoderBlock , ChessPolicyNet , PrepareTorchBaseModel
from Chess.config.configuration import ConfigurationManager
from pathlib import Path
from torch import nn
STAGE_NAME = "PrepareTorchBaseModel"

class PrepareTorchBaseModelPipeline :
    def __init__(self):
        pass

    def main(self) :
        config_manager = ConfigurationManager()
        get_torch_model_config = config_manager.get_torch_base_model()
        torch_base_model = PrepareTorchBaseModel(config=get_torch_model_config)
        torch_base_model.get_base_model()

if __name__ == '__main__':
    try:
        logger.info(f" stage {STAGE_NAME}started ")
        obj = PrepareTorchBaseModelPipeline()
        obj.main()
        logger.info(f" stage {STAGE_NAME} completed ")
    except Exception as e:
        logger.exception(e)
        raise e