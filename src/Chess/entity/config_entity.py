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



