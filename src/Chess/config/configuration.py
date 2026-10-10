
from Chess.utils.common import read_yaml , create_directories
from Chess.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, \
    DataTransformationConfig, PrepareBaseModelConfig, PrepareTrainingConfig, PrepareEvaluationConfig , PrepareEvaluationConfig , PrepareTorchDataset , PrepareTorchBaseModelConfig , PrepareTorchTrainingConfig , PrepareRlModelConfig
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
        config = self.config.evaluate_base_model
        params = self.params

        create_directories([
            config.root_dir
        ])
        prepare_evaluation_config = PrepareEvaluationConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            transformed_x_path=Path(config.transformed_x_path),
            transformed_y_path=Path(config.transformed_y_path),
            evaluate_scores=Path(config.evaluate_scores)

        )
        return prepare_evaluation_config

    def get_torch_dataset (self) -> PrepareTorchDataset:
        config = self.config.torch_dataset
        create_directories([config.root_dir])
        prepare_dataset_config = PrepareTorchDataset(
            root_dir=Path(config.root_dir),
            x_data_path=Path(config.x_data_path),
            y_data_path=Path(config.y_data_path),

        )
        return prepare_dataset_config

    def get_torch_base_model (self) -> PrepareTorchBaseModelConfig :
        config = self.config.torch_base_model
        params = self.params
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareTorchBaseModelConfig(
            root_dir = Path(config.root_dir) ,
            model_path = Path(config.model_path) ,
            params_D_MODEL = params.D_MODEL ,
            params_NUM_HEADS= params.NUM_HEADS ,
            params_FF_DIM = params.FF_DIM ,
            params_IN_CHANNELS= params.IN_CHANNELS ,
            params_NUM_MOVES = params.NUM_MOVES ,
            params_NUM_LAYERS= params.NUM_LAYERS


        )
        return prepare_base_model_config

    def get_torch_training_config(self) -> PrepareTorchTrainingConfig:
        config = self.config.prepare_train_torch_model
        params = self.params
        create_directories([config.root_dir])
        prepare_training_config = PrepareTorchTrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            model_path=Path(config.model_path),
            x_dataset=Path(config.x_dataset),
            y_dataset=Path(config.y_dataset),
            params_epochs=params.EPOCHS,
            params_learning_rate= params.LEARNING_RATE,
            params_batch_size=params.BATCH_SIZE,


        )
        return prepare_training_config
    def get_rl_config(self) -> PrepareRlModelConfig :
        config = self.config.prepare_rl_model
        params = self.params
        create_directories([config.root_dir, config.self_play_data])
        prepare_rl_config = PrepareRlModelConfig(
            root_dir=Path(config.root_dir),
            self_play_data=Path(config.self_play_data),
            params_max_moves=params.MAX_MOVES,
            params_num_games=params.NUM_GAMES,
            params_num_simulation=params.NUM_SIMULATION  ,
            params_learning_rate = float(params.get("RL_LEARNING_RATE", params.LEARNING_RATE)),
            params_epochs= params.EPOCHS ,
            params_batch_size= params.BATCH_SIZE ,
            params_c_puct= params.C_PUCT,
            params_pool_size= int(params.get("POOL_SIZE", 32)),
        )
        return prepare_rl_config