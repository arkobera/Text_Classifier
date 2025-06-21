import os
import zipfile
from pathlib import Path
from src.logger import logging
from src.exception import MyException
from from_root import from_root  # type: ignore


class KaggleConnectionFactory:
    def __init__(self, kaggle_api: str):
        self.kaggle_api = kaggle_api
        self.root = from_root()  # Automatically resolve project root
        logging.info(f"Root directory set to {self.root}")

        self.path = self.root / "data" / "raw"
        os.makedirs(self.path, exist_ok=True)

        self.kaggle_config_dir = self.root / ".kaggle"
        os.environ["KAGGLE_CONFIG_DIR"] = str(self.kaggle_config_dir)

        kaggle_json_path = self.kaggle_config_dir / "kaggle.json"
        if not kaggle_json_path.exists():
            raise MyException("kaggle.json not found", sys) #type: ignore

        logging.info(f"Kaggle config set to: {self.kaggle_config_dir}")

    def download_dataset(self):
        try:
            logging.info(f"Downloading dataset '{self.kaggle_api}' to {self.path}")

            # 🔥 import only after env is set
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(self.kaggle_api, path=self.path)

            logging.info(f"Dataset '{self.kaggle_api}' downloaded successfully to {self.path}")
        except Exception as e:
            raise MyException(e, sys) #type: ignore

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
            raise MyException(e, sys)# type: ignore
