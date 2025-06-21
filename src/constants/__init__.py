import os
import datetime

# Create a timestamp string (e.g., 20240620_153045)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

TARGET = "Survived"
TRAIN_PATH = os.path.join("data", "raw", "train.csv")
TEST_PATH = os.path.join("data", "raw", "test.csv")
PROCESSED_TRAIN_PATH = os.path.join("data", "processed", "train.csv") 
PROCESSED_TEST_PATH = os.path.join("data", "processed", "test.csv")
MODEL_DIR = "models"
