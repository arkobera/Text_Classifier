from src.logger import logging
from src.constants import *
import pandas as pd 
from sklearn.model_selection import KFold
from src.exception import MyException


class Transformation:
    def __init__(self,train,test):
        self.train = train
        self.test = test
        self.target = TARGET
        self.X = self.train.drop(columns=[self.target])
        self.y = self.train[self.target]
        self.strong_cols = ["Sex","Fare"]
        self.weak_cols = ["Embarked","Age","Name"]
        
    # def target_encoding_on_strong_cols(self):
    #     print(f"Starting target encoding on : {self.strong_cols}")
    #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #     for fold,(train_indx,test_indx) in enumerate(kf.split(self.X,self.y)):
    #         print(f"Processing fold {fold+1}")
    #         X_train, X_test = self.X.iloc[train_indx], self.X.iloc[test_indx]
    #         y_train, y_test = self.y.iloc[train_indx], self.y.iloc[test_indx]
    #         for inner_fold,(train_indx,test_indx) in enumerate(kf.split(X_train,y_train)):
    #             print(f"Processing inner fold {inner_fold+1}")
    #             for col in self.strong_cols:
    #                 mean_target = y_train.iloc[train_indx].groupby(X_train[col].iloc[train_indx]).mean()
    #                 X_train[col + f"_target_enc_{fold}_{inner_fold}"] = X_train[col].map(mean_target)
    #                 X_test[col + f"_target_enc_{fold}_{inner_fold}"] = X_test[col].map(mean_target)

        
    def feature_enconding_on_weak_cols(self):
        print(f"Starting feature encoding on : {self.weak_cols}")
        for col1 in self.weak_cols:
            for col2 in self.strong_cols:
                self.train[f"{col1}_{col2}"] = self.train[col1].astype(str) + "_" + self.train[col2].astype(str)
                self.test[f"{col1}_{col2}"] = self.test[col1].astype(str) + "_" + self.test[col2].astype(str)
                df = pd.concat([self.train[f"{col1}_{col2}"],self.test[f"{col1}_{col2}"]],axis=0)
                val,_ = pd.factorize(df) #type: ignore
                self.train[f"{col1}_{col2}"] = val[:len(self.train)]
                self.test[f"{col1}_{col2}"] = val[len(self.train):]
        
    def initiate_feature_transformation(self):
        print("Entering Data Transformation")
        #self.target_encoding_on_strong_cols()
        self.feature_enconding_on_weak_cols()
        return self.train, self.test
    
def main():
    try:
        logging.info("Feature engineering started")
        train = pd.read_csv(PROCESSED_TRAIN_PATH)
        test = pd.read_csv(PROCESSED_TEST_PATH)
        transformer = Transformation(train,test)
        train, test = transformer.initiate_feature_transformation()
        train.to_csv(PROCESSED_TRAIN_PATH, index=False) # type: ignore
        test.to_csv(PROCESSED_TEST_PATH, index=False) # type: ignore    
        logging.info("Feature engineering completed successfully")
    except Exception as e:  
        raise MyException(e, sys) # type: ignore
    

if __name__ == "__main__":
    main()