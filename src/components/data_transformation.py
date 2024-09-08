import sys
from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            categorical_columns = [
                "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod"
            ]
            
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges","SeniorCitizen"]
            
            # Define pipelines for numerical and categorical features
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler())  # MinMaxScaler as requested
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(sparse=False, drop='first')),  # Avoid multicollinearity
                ]
            
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read in data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Data cleaning operations
            logging.info("Performing data cleaning operations")

            # Convert 'TotalCharges' to numeric and handle missing values
            train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
            train_df.dropna(inplace=True)
            train_df.drop('customerID', axis=1, inplace=True)

            test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
            test_df.dropna(inplace=True)
            test_df.drop('customerID', axis=1, inplace=True)

            # Define target and feature columns
            target_column_name = "Churn"
            feature_columns = [col for col in train_df.columns if col != target_column_name]

            # Extract features and target
            X_train_df = train_df[feature_columns]
            y_train_df = train_df[target_column_name].map({'Yes': 1, 'No': 0})

            X_test_df = test_df[feature_columns]
            y_test_df = test_df[target_column_name].map({'Yes': 1, 'No': 0})

            logging.info("Obtaining preprocessing object")

            # Get preprocessing object

            preprocessing_obj = self.get_data_transformer_object()

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing data")

            X_train_arr = preprocessing_obj.fit_transform(X_train_df)
            X_test_arr = preprocessing_obj.transform(X_test_df)
            y_train_arr = np.array(y_train_df)
            y_test_arr = np.array(y_test_df)

            # Save preprocessing object
            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return X_train_arr, X_test_arr, y_train_arr, y_test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)