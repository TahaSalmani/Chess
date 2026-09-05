from Chess.utils.common import read_yaml , create_directories
from Chess.entity.config_entity import  DataIngestionConfig , DataValidationConfig
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


