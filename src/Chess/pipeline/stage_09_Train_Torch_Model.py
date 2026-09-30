from Chess import logger
from Chess.components.train_torch_base_model import TrainTorchBaseModel
from Chess.config.configuration import ConfigurationManager

STAGE_NAME = "TorchModelTraining"

class TrainTorchModelPipeline :
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()

        get_torch_model_train = config_manager.get_torch_training_config()
        base_model_config = config_manager.get_torch_base_model()

        trainer = TrainTorchBaseModel(
            config=get_torch_model_train,
            base_model_config=base_model_config
        )
        trainer.train()

if __name__ == "__main__":
    logger.info("Training Torch Model Started")
    obj = TrainTorchModelPipeline()
    obj.main()
    logger.info("Training Torch Model Finished")

