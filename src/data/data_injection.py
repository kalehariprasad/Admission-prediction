import pandas as pd
import numpy as np
import os 
import sys
from src.Exception.exception import CustomException
from src.logger.my_logging import logger
from sklearn.model_selection import train_test_split
from src.utils.utilities import save_data


class DataInjection:
    def __init__(self) -> None:
        pass
    def initiate_data_injection(self):
        try:
            data_dir="data/raw/data.csv"
            df=pd.read_csv(data_dir)
            x_train,y_train=train_test_split(df)
            logger.info('data spliting compleeted')
            train_path = "data/external/train.csv"  
            test_path = "data/external/test.csv"    
            save_data(x_train, train_path)
            logger.info('Training data stored successfully')
            save_data(y_train, test_path)
            logger.info('Testing data stored successfully')
            return (x_train,y_train)
        except Exception as e:
                logger.error(f"error occured during data_injection with {e}")
                raise CustomException(e,sys)     


if __name__=="__main__":
     obj=DataInjection()
     obj.initiate_data_injection()