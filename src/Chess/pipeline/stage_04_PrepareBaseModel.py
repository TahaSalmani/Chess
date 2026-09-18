from src.Chess import logger
from src.Chess.components.prepare_base_model import PrepareBaseModel
from src.Chess.config.configuration import ConfigurationManager

class PrepareBaseModelPipeline:
    def __init__(self):
        pass
    def main (self):
        config = ConfigurationManager()
        prepare_base_model_config   = config.get_base_model()
        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_model()



STAGE_NAME = "PrepareBaseModelPipeline"

if __name__ == "__main__":
    try :
        logger.info (f">>>>>>>>>> stage {STAGE_NAME} Started")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info (f">>>>>>>>>>> stage {STAGE_NAME} Finished")
    except Exception as e :
        logger.exception(e)
        raise e
