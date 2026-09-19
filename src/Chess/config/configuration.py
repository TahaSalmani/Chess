from Chess.utils.common import read_yaml , create_directories
from Chess.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, \
    DataTransformationConfig, PrepareBaseModelConfig, PrepareTrainingConfig, PrepareEvaluationConfig , PrepareEvaluationConfig
from Chess.constants import Params_File_Path , Config_File_Path
from pathlib import Path

class ConfigurationManager :
    def __init__(self  ,
    config_file_path = Config_File_Path ,
    params_file_path = Params_File_Path ) :
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig :
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_file =config.source_file,
            local_data_file = config.local_data_file,
            unzip_dir = Path(config.unzip_dir),
            source_URL=config.source_URL,
        )
        return data_ingestion_config

    def get_data_validation_config(self)-> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            source_file=Path(config.source_file),
            root_dir=Path(config.root_dir),
            status_file = Path(config.status_file)
        )
        return data_validation_config


    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir = Path(config.root_dir),
            data_path=Path(config.data_path),
            status_file = Path(config.status_file) ,
            transformed_data_dir = Path(config.transformed_data_dir),
        )
        return data_transformation_config

    def get_base_model(self)-> PrepareBaseModelConfig :
        config = self.config.prepare_base_Model
        params = self.params
        create_directories([config.root_dir])

        base_model_config = PrepareBaseModelConfig(
            model_path=Path(config.model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES ,
            root_dir=Path(config.root_dir),
            updated_base_model_path=Path(config.updated_base_model_path),

        )
        return base_model_config

    def get_train_model(self) -> PrepareTrainingConfig:
        config = self.config.prepare_train_model
        params = self.params

        create_directories([Path(config.root_dir)])

        prepare_train_config = PrepareTrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            transformed_x_path=Path(config.transformed_x_path),
            transformed_y_path=Path(config.transformed_y_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_shuffle=params.SHUFFLE,
            params_validation_split=params.VALIDATION_SPLIT,
            params_learning_rate=params.LEARNING_RATE,
        )

        return prepare_train_config

    def get_evaluation_model(self)-> PrepareEvaluationConfig:
        config = self.config.prepare_evaluation_model
        params = self.params

        create_directories([
            config.root_dir,
            Path(config.evaluated_model_path).parent,
            config.evaluation_Unzip_Data,
            config.evaluate_Downloaded_Data,
            config.evaluate_validation_dir,
            config.evaluate_transformation ,
            config.evaluate_scores
        ])
        prepare_evaluation_config = PrepareEvaluationConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            evaluated_model_path=Path(config.evaluated_model_path),

            source_file =config.source_file,
            local_data_file = config.local_data_file,
            unzip_dir = Path(config.unzip_dir),
            source_URL=config.source_URL,
            status_file=Path(config.status_file) ,

            data_path=Path(config.data_path),
            transformed_data_dir=Path(config.transformed_data_dir),

            evaluate_scores = Path(config.evaluate_scores),

        )
        return prepare_evaluation_config