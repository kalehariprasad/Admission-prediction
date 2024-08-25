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
        #print(f"Saving array to: {file_path}")
    except Exception as e:
        logger.error(f"error occured while saving numpy array with {e}")
        raise CustomException(e,sys)
    
def load_array(file_path):
    try:
        # Check if the file exists before trying to load
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Load the NumPy array from the file
        array = np.load(file_path)
        return array
    except Exception as e:
        logger.error(f"Error occurred while loading numpy array: {e}")
        raise CustomException(e, sys)
    
def save_object(obj,file_path):
    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        joblib.dump(obj, file_path)
        
    except Exception as e:
        logger.error(f"error occured while saving joblib with {e}")
        raise CustomException(e,sys)


def train_model(model,x_train,y_train):
    try:
        model.fit(x_train,y_train)
        return model
    except Exception as e:
        logger.error(f"error occuered while training the model with {e}")
        raise CustomException(e,sys)
    
