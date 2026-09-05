from Chess import logger
from src.Chess.entity.config_entity import DataValidationConfig
from src.Chess.components.data_validation import DataValidation
from src.Chess.config.configuration import ConfigurationManager

STAGE_NAME = "DataValidation"
class DataValidationPipeline:
    def __init__(self):
        pass
    def main(self ):
        config = ConfigurationManager()
        validation_config = config.get_data_validation_config()
        validation_data = DataValidation(validation_config)
        validation_data.validate()

if __name__ == "__main__":
    logger.info(f">>>>>> stage {STAGE_NAME} started")
    try:

        obj = DataValidationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed")


    except Exception as e :
        logger.error(f">>>>>> stage {STAGE_NAME} failed: {e}")
        logger.exception(e)
        raise e
