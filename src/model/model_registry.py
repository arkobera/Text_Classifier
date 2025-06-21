from src.logger import logging
from src.exception import MyException
from src.constants import *
import mlflow
import dagshub
import json

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri('https://dagshub.com/arkobera/Titanic_Survival_Prediction.mlflow')
dagshub.init(repo_owner='arkobera', repo_name='Titanic_Survival_Prediction', mlflow=True) #type: ignore

class ModelRegistry:

    def __init__(self,file_path,model_name):
        self.file_path = file_path
        self.model_name = model_name

    def load_model_info(self):
        try:
            with open(self.file_path, 'r') as file:
                self.model_info = json.load(file)
            logging.debug('Model info loaded from %s', self.file_path)
        except Exception as e:
            raise MyException(e,sys) #type: ignore
        
    def register_model(self):
        try:
            model_uri = f"runs:/{self.model_info['run_id']}/{self.model_info['model_path']}"
            # Register the model
            model_version = mlflow.register_model(model_uri, self.model_name)
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()  #type: ignore
            client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Staging"
                )
            logging.debug(f'Model {self.model_name} version {model_version.version} registered and transitioned to Staging.')
        except Exception as e:
            raise MyException(e,sys) #type: ignore
        
def main():
    try:
        model_path = "reports/exp_info.json"
        registry = ModelRegistry(model_path,"my_model")
        registry.load_model_info()
        registry.register_model()
    except Exception as e:
        raise MyException(e,sys) #type: ignore
    
if __name__ == "__main__":
    main()
