import os
import sys
import pandas as pd
import numpy as np
from src.Exception.exception import CustomException
from src.logger.my_logging import logger
import joblib


def save_data(data: pd.DataFrame, file_path: str):
    try:
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        
        data.to_csv(file_path, index=False)  
        logger.info('Data saved successfully')
    except Exception as e:
        logger.error(f"Error occurred during saving data: {e}")
        raise CustomException(e, sys)

def save_array(array,file_path):
    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        np.save(file_path,array)
    except Exception as e:
        logger.error(f"error occured while saving numpy array with {e}")
        raise CustomException(e,sys)
    
def save_object(obj,file_path):
    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        joblib.dump(obj, file_path)
        
    except Exception as e:
        logger.error(f"error occured while saving joblib with {e}")
        raise CustomException(e,sys)
