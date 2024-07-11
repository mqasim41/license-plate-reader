from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    source_URL: str
    mlflow_uri: str
    base_model_path: Path
    updated_base_model_path: Path
    

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    base_model_path: Path
    mlflow_uri: str
    format: str
    params_epochs: int
    params_image_size: list
    params_resume: bool


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path

