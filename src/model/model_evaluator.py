from src.logger import logging
from src.exception import MyException
from src.constants import *         
from sklearn.metrics import * # type: ignore
from xgboost import XGBClassifier
from src.Artifacts import *
from sklearn.model_selection import train_test_split
import pickle
import sys
import pandas as pd
import numpy as np
import pickle
import json
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os

mlflow.set_tracking_uri('https://dagshub.com/arkobera/Titanic_Survival_Prediction.mlflow')
dagshub.init(repo_owner='arkobera', repo_name='Titanic_Survival_Prediction', mlflow=True) #type: ignore



class ModelEvaluator:
    def __init__(self):
        self.eval_dir = os.path.join(file_paths.EVALUATION_DIR)
        os.makedirs(self.eval_dir,exist_ok=True)
        self.modelartifacts = ModelArtifacts()
        self.file_paths = file_paths()

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.file_paths.MODEL_TEST_AND_PRED_DIR}")
            # Load your data here
            self.X_test = pd.read_csv(os.path.join(self.file_paths.MODEL_TEST_AND_PRED_DIR,"x_test.csv"))
            self.y_test = pd.read_csv(os.path.join(self.file_paths.MODEL_TEST_AND_PRED_DIR,"y_test.csv"))
            logging.info("Data loaded successfully.")
        except Exception as e:
            raise MyException(e, sys)  #type: ignore
        
    def load_model(self):
        try:
            model_path = self.modelartifacts.MODEL_TRAINED_PATH
            logging.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    def evaluate_model(self):
        try:
            logging.info("Evaluating Data")
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            self.metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
            }
            logging.info('Model evaluation metrics calculated')
        except Exception as e:
            raise MyException(e,sys) #type: ignore
        
    def save_metrics(self):
        try:
            file_path = os.path.join(self.eval_dir,"RES.json")
            with open(file_path, 'w') as file:
                json.dump(self.metrics_dict, file, indent=4)
            logging.info('Metrics saved to %s', file_path)
        except Exception as e:
            raise MyException(e,sys) #type: ignore
    
    def save_model_info(self,run_id):
        try:
            file_path = os.path.join(self.eval_dir,"exp_info.json")
            model_info = {"run_id":run_id,"model_path":self.modelartifacts.MODEL_TRAINED_PATH}
            with open(file_path,"w") as file:
                json.dump(model_info,file,indent=4)
            logging.debug('Model info saved to %s', file_path)
        except Exception as e:
            raise MyException(e,sys) #type: ignore
        
def main():
    mlflow.set_experiment("dvc_pipeline")
    with mlflow.start_run() as run:
        try:
            evaluator = ModelEvaluator()
            evaluator.load_data()
            evaluator.load_model()
            evaluator.evaluate_model()
            evaluator.save_metrics()
            for metric_name,metric_value in evaluator.metrics_dict.items():
                mlflow.log_metric(metric_name,metric_value) #type: ignore
            if hasattr(evaluator.model,"get_params"):
                params = evaluator.model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name,param_value)
            mlflow.sklearn.log_model(evaluator.model,"model") #type: ignore
            evaluator.save_model_info(run.info.run_id)
            mlflow.log_artifact(os.path.join(evaluator.eval_dir,"RES.json"))
        except Exception as e:
            raise MyException(e,sys) #type: ignore
    
if __name__ == "__main__":
    main()



