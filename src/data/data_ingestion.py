# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from src.connections.kaggle_connection import KaggleConnectionFactory
from src.exception import MyException

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logging.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logging.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logging.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error: %s', e)
#         raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except Exception as e:
        raise MyException(e,sys) #type: ignore

# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess the data."""
#     try:
#         # df.drop(columns=['tweet_id'], inplace=True)
#         logging.info("pre-processing...")
#         final_df = df[df['sentiment'].isin(['positive', 'negative'])]
#         final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
#         logging.info('Data preprocessing completed')
#         return final_df
#     except KeyError as e:
#         logging.error('Missing column in the dataframe: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error during preprocessing: %s', e)
#         raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        raise MyException(e,sys) #type: ignore

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        
        # df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")
        


        # final_df = preprocess_data(df)
        # train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        # save_data(train_data, test_data, data_path='./data')
        #print("Imported object type:", type(KaggleConnectionFactory))
        competition_name = "titanic"  # ✅ just the competition name
        kaggle_conn = KaggleConnectionFactory(competition_name)
        kaggle_conn.download_dataset()
        logging.info(f"Dataset name: {competition_name}")
        kaggle_conn.unzip_dataset(zip_file=f"{competition_name}.zip")

    except Exception as e:
        raise MyException(e,sys) #type: ignore

if __name__ == '__main__':
    main()