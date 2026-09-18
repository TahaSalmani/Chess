from Chess import logger
from Chess.config.configuration import ConfigurationManager
from Chess.components.train_base_model import TrainBaseModel


class TrainModelPipeline :
    def __init__(self):
        pass
    def main(self ):
        config = ConfigurationManager()
        train_model_config = config.get_train_model()
        train_model =  TrainBaseModel(train_model_config)
        train_model.Train_Base_Model()

STAGE_NAME = "train_model"

if __name__ == "__main__":

    try:
        logger.info(f"stage{STAGE_NAME} started")
        obj = TrainModelPipeline()
        obj.main()
        logger.info(f"stage{STAGE_NAME} completed")

    except Exception as e:
        logger.info(f"stage{STAGE_NAME} failed")
        raise e

