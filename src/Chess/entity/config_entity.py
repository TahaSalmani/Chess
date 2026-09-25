from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig :
    root_dir: Path
    source_file: Path
    local_data_file : Path
    source_URL : str
    unzip_dir : Path

@dataclass(frozen=True)
class DataValidationConfig :
    root_dir: Path
    source_file: Path
    status_file : Path
@dataclass(frozen=True)
class DataTransformationConfig :
    root_dir: Path
    data_path: Path
    status_file : Path
    transformed_data_dir : Path
@dataclass(frozen=True)
class PrepareBaseModelConfig :
    root_dir: Path
    model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_classes: int
    updated_base_model_path: Path
@dataclass(frozen=True)
class PrepareTrainingConfig :
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    transformed_x_path: Path
    transformed_y_path: Path
    params_epochs: int
    params_batch_size: int
    params_shuffle: bool
    params_validation_split: float
    params_learning_rate: float
@dataclass(frozen=True)
class PrepareEvaluationConfig :
    root_dir: Path
    trained_model_path: Path
    transformed_x_path: Path
    transformed_y_path: Path
    evaluate_scores: Path
@dataclass(frozen=True)
class PrepareRlEnvConfig :
    board_cols : int
    board_rows : int
    pieces_type : int
    action_space_size : int
    reward : dict
@dataclass(frozen=True)
class PrepareRlLearnConfig :
    verbose : bool
    learning_rate : float
    total_timesteps : int
    root_dir : Path
    tensorboard_log_dir: Path

