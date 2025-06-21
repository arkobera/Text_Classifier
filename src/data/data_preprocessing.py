from src.logger import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.constants import *
from src.exception import MyException

class Preprocessor:
    def __init__(self,train,test):
        self.train = train
        self.test = test
        
    def set_target(self):
        # y_true = self.train[TARGET]
        self.train.drop(columns=["PassengerId"],inplace=True)
        self.test.drop(columns=["PassengerId"],inplace=True)
        self.df = pd.concat([self.train,self.test],axis=0)
        self.obj_col = [col for col in self.test if self.test[col].dtype == "object"]
        self.num_col = [col for col in self.test if self.test[col].dtype in ["int64","float64"] and col != TARGET]
        
    def impute_obj(self):
        for col in self.obj_col:
            val,_ = pd.factorize(self.df[col])
            self.train[col] = val[:len(self.train)]
            self.test[col] = val[len(self.train):]
            
    def scale_num_col(self):
        for col in self.num_col:
            scaler = StandardScaler()
            self.train[col] = scaler.fit_transform(self.train[[col]])
            self.test[col] = scaler.transform(self.test[[col]])

    def fill_nan(self):
        self.train.fillna(0,inplace = True)
        self.test.fillna(0,inplace = True)
            
    def initiate_preprocessing(self):
        self.set_target()
        self.impute_obj()
        self.scale_num_col()
        self.fill_nan()
        return self.train,self.test
    
def main():
    try:
        os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(PROCESSED_TEST_PATH), exist_ok=True)
        logging.info("Preprocessing started")
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        preprocessor = Preprocessor(train,test)
        train,test = preprocessor.initiate_preprocessing()
        train.to_csv(PROCESSED_TRAIN_PATH,index=False)
        test.to_csv(PROCESSED_TEST_PATH,index=False)
        logging.info("Preprocessing completed successfully")
    except Exception as e:
        raise MyException(e,sys) #type: ignore
    
if __name__ == "__main__":
    main()