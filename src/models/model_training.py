import pandas as pd
import numpy as np
import os
import sys
from src.Exception.exception import CustomException
from src.logger.my_logging import logger
from src.utils.utilities import  load_array, save_object, train_model
from sklearn.ensemble import RandomForestClassifier

train_x_path = "data/processed/train_x.npy"
train_y_path = "data/processed/train_y.npy"
test_x_path = "data/processed/test_x.npy"
test_y_path = "data/processed/test_y.npy"


def model_training():
    try:
        logger.debug('Model training started')
        model = RandomForestClassifier()
        train_x = load_array(train_x_path)
        train_y = load_array(train_y_path)
        test_x = load_array(test_x_path)
        test_y = load_array(test_y_path)
        model = train_model(model=model, x_train=train_x, y_train=train_y)
        pred=model.predict(test_x)
        return model
    except Exception as e:
        logger.error(f"Error occurred in model_training.py file with {e}")
        raise CustomException(e, sys)
    
    
if __name__ == "__main__":
    model=model_training()
    model_path="models/model.pkl"
    save_object(model,model_path)
