from src.logger import logging
from src.exception import MyException
from src.constants import *         
from sklearn.metrics import * # type: ignore
from xgboost import XGBClassifier
from src.Artifacts import ModelArtifacts,file_paths
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

class ModelBuilder:
    def __init__(self, model_params,model_name):
        self.modelartifacts = ModelArtifacts()
        self.file_paths = file_paths()
        self.model_params = model_params
        self.model_name = model_name

    def load_data(self):
        try:
            logging.info(f"Loading data from {PROCESSED_TRAIN_PATH} and {PROCESSED_TEST_PATH}")
            # Load your data here
            self.train = pd.read_csv(PROCESSED_TRAIN_PATH)
            self.X = self.train.drop(columns=[TARGET])
            self.y = self.train[TARGET]
            logging.info("Data loaded successfully.")
        except Exception as e:
            raise MyException(e, sys) #type: ignore

    def split_data(self, n_splits=5):
        try:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logging.info(f"Data split into training and testing sets with test size: 0.2")
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    def save_test_and_prediction_data(self):
        try:
            logging.info(f"Saving test and prediction data to {self.file_paths.MODEL_TEST_AND_PRED_DIR}")
            os.makedirs(self.file_paths.MODEL_TEST_AND_PRED_DIR,exist_ok=True)
            self.x_test.to_csv(os.path.join(self.file_paths.MODEL_TEST_AND_PRED_DIR,"x_test.csv"), index=False)
            self.y_test.to_csv(os.path.join(self.file_paths.MODEL_TEST_AND_PRED_DIR,"y_test.csv"), index=False)
            logging.info("Test and prediction data saved successfully.")
        except Exception as e:
            raise MyException(e, sys) #type: ignore
    
    def train_xgb_classifier(self, X_train, y_train):
        try:
            logging.info(f"Training {self.model_name} with parameters: {self.model_params}")
            self.model = XGBClassifier(**self.model_params)
            self.model.fit(X_train, y_train)
            logging.info(f"{self.model_name} training completed.")
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    # def evaluate_model(self, X_test, y_test):
    #     try:
    #         logging.info(f"Evaluating {self.model_name}")
    #         y_pred = self.model.predict(X_test)
    #         accuracy = accuracy_score(y_test, y_pred)
    #         precision = precision_score(y_test, y_pred, average='weighted')
    #         recall = recall_score(y_test, y_pred, average='weighted')
    #         f1 = f1_score(y_test, y_pred, average='weighted')
    #         logging.info(f"{self.model_name} evaluation completed with accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
    #         return {
    #             'accuracy': accuracy,
    #             'precision': precision,
    #             'recall': recall,
    #             'f1': f1
    #         }
        
        # except Exception as e:
        #     raise MyException(e, sys) #type: ignore
    
    def save_model(self):
        try:
            model_dir = os.path.join(MODEL_DIR)
            os.makedirs(model_dir,exist_ok=True)
            model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
            os.makedirs(MODEL_DIR, exist_ok=True) 
            self.modelartifacts.MODEL_TRAINED_PATH = model_path  # type: ignore
            logging.info(f"Saving model to {model_path}")
            logging.info(f"Loggin inof updated {self.modelartifacts.MODEL_TRAINED_PATH}")
            with open(model_path, 'wb') as model_file:
                pickle.dump(self.model, model_file)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            raise MyException(e, sys) #type: ignore
    
    # def load_model(self):
    #     try:
    #         model_path = self.modelartifacts.MODEL_TRAINED_PATH  # type: ignore
    #         logging.info(f"Loading model from {model_path}")
    #         with open(model_path, 'rb') as model_file:
    #             self.model = pickle.load(model_file)
    #         logging.info(f"Model loaded successfully from {model_path}")
    #     except Exception as e:
    #         raise MyException(e, sys) #type: ignore
        
    def predict(self, X):
        try:
            logging.info(f"Making predictions with {self.model_name}")
            predictions = self.model.predict(X)
            logging.info(f"Predictions completed.")
            return predictions
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
    def build(self):
        try:
            logging.info(f"Building and evaluating model: {self.model_name}")
            self.load_data()
            self.split_data()
            self.save_test_and_prediction_data()
            self.train_xgb_classifier(self.x_train, self.y_train)
            # evaluation_results = self.evaluate_model(self.x_test, self.y_test)
            self.save_model()
            logging.info(f"Model building and evaluation completed successfully.")
        except Exception as e:
            raise MyException(e, sys) #type: ignore
        
def main():
    # Example usage
    model_params = {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.01,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    model_name = "XGBClassifier"
    
    # Assuming X and y are defined
    # X, y = load_your_data()  # Replace with actual data loading logic
    
    model_builder = ModelBuilder(model_params, model_name)
    model_builder.build()

if __name__ == "__main__":
    main()