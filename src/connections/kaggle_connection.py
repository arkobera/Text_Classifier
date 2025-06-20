import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

from src.logger import logging
from from_root import from_root  # type: ignore


class KaggleConnectionFactory:
    def __init__(self, kaggle_api: str):
        self.kaggle_api = kaggle_api

        # Automatically resolve project root
        self.root = from_root()  # type: ignore
        logging.info(f"Root directory set to {self.root}")

        # Define dataset path
        self.path = self.root / "data" / "raw"
        os.makedirs(self.path, exist_ok=True)

        # Set Kaggle config dir to local project `.kaggle` folder
        self.kaggle_config_dir = self.root / ".kaggle"
        os.environ["KAGGLE_CONFIG_DIR"] = str(self.kaggle_config_dir)

        # Confirm kaggle.json exists
        kaggle_json_path = self.kaggle_config_dir / "kaggle.json"
        if not kaggle_json_path.exists():
            raise FileNotFoundError(f"kaggle.json not found at: {kaggle_json_path}")
        logging.info(f"Kaggle config set to: {self.kaggle_config_dir}")

    def download_dataset(self):
        try:
            logging.info(f"Downloading dataset '{self.kaggle_api}' to {self.path}")
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(self.kaggle_api, path=self.path)
            logging.info(f"Dataset '{self.kaggle_api}' downloaded successfully to {self.path}")
        except Exception as e:
            logging.error(f"Error downloading dataset '{self.kaggle_api}': {e}")
            raise

    def unzip_dataset(self, zip_file: str, overwrite: bool = True):
        try:
            zip_path = self.path / zip_file

            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found at: {zip_path}")

            logging.info(f"Unzipping dataset {zip_path} into {self.path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            logging.info(f"Dataset {zip_file} unzipped successfully into {self.path}")
        except Exception as e:
            logging.error(f"Error unzipping dataset {zip_file}: {e}")
            raise
