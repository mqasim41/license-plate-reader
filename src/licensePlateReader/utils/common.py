import os
from box.exceptions import BoxValueError
import yaml
from licensePlateReader import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import gdown
import zipfile
import mlflow
from ultralytics import YOLO
import torch

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    

def download_file(source_url, local_data_file, root_dir):
    '''
    Fetch data from the url
    '''

    try: 
        dataset_url = source_url
        zip_download_dir = local_data_file
        os.makedirs(root_dir, exist_ok=True)
        logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

        file_id = dataset_url.split("/")[-2]
        prefix = 'https://drive.google.com/uc?/export=download&id='
        gdown.download(prefix+file_id,zip_download_dir)

        logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

    except Exception as e:
        raise e

def extract_zip_file(unzip_dir, local_data_file):
    """
    zip_file_path: str
    Extracts the zip file into the data directory
    Function returns None
    """
    unzip_path = unzip_dir
    os.makedirs(unzip_path, exist_ok=True)
    with zipfile.ZipFile(local_data_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)

def download_mlflow_artifact(run_id, artifact_path='weights/best.pt', local_dir=None):
    """
    Downloads an artifact from an MLflow run to a specified local directory.

    Parameters:
    - run_id: str, the ID of the MLflow run.
    - artifact_path: str, the relative path to the artifact within the run.
    - local_dir: str, optional, the local directory where the artifact should be downloaded. If not specified, the artifact will be downloaded to the current working directory.

    Returns:
    - local_path: str, the local path where the artifact was downloaded.
    """
    if local_dir:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=local_dir)
    else:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)


def load_yolo_model(model_path, device='cpu'):
    """
    Load YOLO model from the specified path and move it to the specified device (cpu or cuda).
    """
    model = YOLO(model_path)
    if device == 'cuda' and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model


def update_yaml(file_path, key1,key2, value):
    
    with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Update the key with the current value
    data[key1][key2] = value

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

    

