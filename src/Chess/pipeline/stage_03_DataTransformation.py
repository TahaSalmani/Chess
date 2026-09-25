from Chess import logger
from Chess.components.data_transformation import DataTransformation
from Chess.entity.config_entity import DataTransformationConfig
from Chess.config.configuration import ConfigurationManager
import chess

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config = transformation_config)
        data_transformation.initiate_data_transformation()





if __name__ == "__main__":
    try :
        logger.info("TransFormation Started .....")
        obj =DataTransformationPipeline()
        obj.main()
        logger.info("TransFormation Finished .....")
    except Exception as e :
        logger.error("TransFormation Exception : {}".format(e))
        raise e


