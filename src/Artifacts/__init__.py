import os
from dataclasses import dataclass


class file_paths:
    PROCESSED_TRAIN_PATH = "data/processed/train.csv"
    PROCESSED_TEST_PATH = "data/processed/test.csv"
    MODEL_DIR = "models/"
    MODEL_TEST_AND_PRED_DIR = "data/model_test_and_prediction/"
    TRAIN_PATH = "data/raw/train.csv"
    TEST_PATH = "data/raw/test.csv"
    TARGET = "Survived"  # Example target column, adjust as needed
    TIMESTAMP = "20231001_120000"  # Example timestamp, adjust as needed
    EVALUATION_DIR = "reports"

class ModelArtifacts:
    MODEL_TRAINED_PATH = "models/XGBClassifier.pkl"

