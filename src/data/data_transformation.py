import pandas as pd
import numpy as np
import os 
import sys
from src.Exception.exception import CustomException
from src.logger.my_logging import logger
from src.utils.utilities import save_data, save_array,save_object
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class CustomPipelineCreator(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable='in_college'):
        self.target_variable = target_variable
    
    def create_pipelines(self):
        """
        Create pipelines for each type of feature.
        """
        try:

            # Define individual pipelines
            logger.debug('started creating piplines for each column')
            self.type_school_pipe = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))  # Adjusted for sparse matrices
            ])

            self.school_accreditation_pipe = Pipeline(steps=[
                ("encoder", OrdinalEncoder(categories=[['A', 'B']]))
            ])

            self.gender_pipe = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))  # Adjusted for sparse matrices
            ])

            self.interest_pipe = Pipeline(steps=[
                ("encoder", OrdinalEncoder(categories=[
                    ['Not Interested', 'Less Interested', 'Very Interested', 'Uncertain', 'Quiet Interested']
                ])),
                ("scaler", StandardScaler(with_mean=False))  # Adjusted for sparse matrices
            ])

            self.residence_pipe = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))  # Adjusted for sparse matrices
            ])

            self.parent_age_pipe = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=True))  # StandardScaler works fine with dense data
            ])

            self.parent_salary_pipe = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=True))  # StandardScaler works fine with dense data
            ])

            self.average_grades_pipe = Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=True))  # StandardScaler works fine with dense data
            ])

            self.parent_in_college_pipe = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))  # Adjusted for sparse matrices
            ])
            logger.debug('compleeted creating piplines for each column')
        except Exception as e:
                logger.error(f"error occured while crteating pipeline with {e}")
                raise CustomException(e,sys)   
    
    def create_preprocessor(self):
        """
        Create and return the ColumnTransformer using the pipelines defined in create_pipelines.
        """
        try:

            self.create_pipelines()  # Initialize pipelines
            logger.debug('created create_pipeline function instance and started combing piplines with columTransformer')
            preprocessor = ColumnTransformer(
                transformers=[
                    ("type_school", self.type_school_pipe, ['type_school']),
                    ("school_accreditation", self.school_accreditation_pipe, ['school_accreditation']),
                    ("gender", self.gender_pipe, ['gender']),
                    ("interest", self.interest_pipe, ['interest']),
                    ("residence", self.residence_pipe, ['residence']),
                    ("parent_age", self.parent_age_pipe, ['parent_age']),
                    ("parent_salary", self.parent_salary_pipe, ['parent_salary']),
                    ("average_grades", self.average_grades_pipe, ['average_grades']),
                    ("parent_in_college", self.parent_in_college_pipe, ['parent_was_in_college'])
                ],
                remainder='passthrough'  # This will pass through the columns not explicitly mentioned in transformers
            )
            logger.debug('combined all pipelines as preprocesspor using Columntransformer')
            return preprocessor
        except Exception as e:
                logger.error(f"error occured  while creating preprocessor with {e}")
                raise CustomException(e, sys)   

    def remove_target_variable(self, X):
        """
        Remove the target variable from the dataframe if it is set.
        """
        try:
            self.target_variable and self.target_variable in X.columns
            #logger.debug('spliting  X and Y Variables from data frame')
            y = X[self.target_variable] # Convert target column to numpy array
            X = X.drop(columns=[self.target_variable])
            #logger.debug('splitted X and Y variables')
            return (X,y)
        except Exception as e:
                logger.error(f"error occured  while removing target column with {e}")
                raise CustomException(e, sys)   
    
    def intiate_preprocessing(self, train_df, test_df):
        """
        Initialize preprocessing: remove target variable and apply the preprocessing pipeline.
        """
        try:
            logger.debug('preprocessing started ')
            train_x,train_y = self.remove_target_variable(train_df)
            logger.debug('X and Y variables splitted for train_df')
            test_x,test_y = self.remove_target_variable(test_df)
            logger.debug('X and Y variables splitted for test_df')
            preprocessor = self.create_preprocessor()
            train_x = preprocessor.fit_transform(train_x)
            logger.debug('applied preprocessing on train_x')
            #preprocessor saving
            path="models/objects/preprocessor.pkl"
            save_object(preprocessor,path)
            logger.info(f"preprocessor saved in {path} ")
            test_x=preprocessor.transform(test_x)
            logger.debug('applied preprocessing on test_x')
            return (train_x, train_y, test_x, test_y)
        except Exception as e:
                logger.error(f"error occured during preprocessing with {e}")
                raise CustomException(e, sys)   

    
if __name__=='__main__':
     train_path = "data/external/train.csv"  
     test_path = "data/external/test.csv"
     train_x_path = "data/processed/train_x.npy"
     train_y_path = "data/processed/train_y.npy"
     test_x_path = "data/processed/test_x.npy"
     test_y_path = "data/processed/test_y.npy"
     train_data=pd.read_csv(train_path)
     test_data=pd.read_csv(test_path)
     obj= CustomPipelineCreator()
     train_x,train_y,test_x,test_y=obj.intiate_preprocessing(train_data, test_data)
     save_array(train_x, train_x_path)  
     save_array(train_y, train_y_path)
     save_array(test_x, test_x_path)
     save_array(test_y, test_y_path)