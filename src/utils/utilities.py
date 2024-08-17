import os
import sys
import pandas as pd
from src.Exception.exception import CustomException
from src.logger.my_logging import logger

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
