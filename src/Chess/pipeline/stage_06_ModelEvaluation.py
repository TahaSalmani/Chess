
from Chess.config.configuration import ConfigurationManager
from Chess.components.data_ingestion import DataIngestion
from Chess.components.data_validation import DataValidation
from Chess.components.data_transformation import DataTransformation
from Chess.components.model_evaluate import ModelEvaluate
from Chess import logger

STAGE_NAME = "Model Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()



        logger.info("Evaluating Model...")
        eval_config = config_manager.get_evaluation_model()
        model_evaluator = ModelEvaluate(config=eval_config)
        model_evaluator.evaluation()
        model_evaluator.save_score()


if __name__ == '__main__':
    try:
        logger.info(f"stage {STAGE_NAME} started")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f" stage {STAGE_NAME} completed ")
    except Exception as e:
        logger.exception(e)
        raise e