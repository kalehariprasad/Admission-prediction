import os
import sys
from src.Exception.exception import CustomException
from src.logger.my_logging import logger
from src.utils.utilities import  load_array, load_object
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


test_x_path = "data/processed/test_x.npy"
test_y_path = "data/processed/test_y.npy"
model_path="models/model.pkl"


def model_evalution():
    try:
        logger.debug('Model Evaluation started')
        
        test_x = load_array(test_x_path)
        test_y = load_array(test_y_path)
        model=load_object(model_path)
        pred=model.predict(test_x)
        conf_matrix = confusion_matrix(test_y, pred)
        acc = accuracy_score(test_y, pred)
        prec = precision_score(test_y, pred) 
        rec = recall_score(test_y, pred)      

        logger.debug(f"Metrics for model are \n"
            f"Confusion Matrix:\n{conf_matrix}\n"
            f"Accuracy: {acc:.2f}\n"
            f"Precision: {prec:.2f}\n"
            f"Recall: {rec:.2f}")
        
        return model
    except Exception as e:
        logger.error(f"Error occurred in model_training.py file with {e}")
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    model_evalution()