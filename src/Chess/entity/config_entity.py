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


