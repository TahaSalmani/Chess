# src/Chess/pipeline/stage_06_ModelEvaluation.py

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

        logger.info("Executing Data Ingestion for Evaluation...")
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        extracted_file_path = data_ingestion.extract_zst_file()

        logger.info("Executing Data Validation...")
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        is_valid = data_validation.validate()

        if not is_valid:
            raise Exception("Data validation failed! Stopping evaluation pipeline.")

        logger.info("Transforming Data to NumPy format...")
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        X_test, y_test = data_transformation.transform_pgn_to_numpy(extracted_file_path)

        logger.info("Evaluating Model...")
        eval_config = config_manager.get_evaluation_model()
        model_evaluator = ModelEvaluate(config=eval_config)
        model_evaluator.evaluation(X_test, y_test)


if __name__ == '__main__':
    try:
        logger.info(f"stage {STAGE_NAME} started")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f" stage {STAGE_NAME} completed ")
    except Exception as e:
        logger.exception(e)
        raise e