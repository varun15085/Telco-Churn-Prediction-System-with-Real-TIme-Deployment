import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scipy.stats import uniform, randint

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from scipy.stats import uniform, randint

from exception import CustomException
from logger import logging
from utils import save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train,X_test, y_train,y_test):
        try:
            logging.info("Split training and test input data")
            # X_train, y_train, X_test, y_test = (
            #     train_array[:, :-1],
            #     train_array[:, -1],
            #     test_array[:, :-1],
            #     test_array[:, -1]
            # )
            # print("XTRAIN SHAPE : INSIDE MODEL TRAINER")  
            # print(X_train.shape)

            # Initialize an empty list to store model scores
            model_scores = []
            best_model_name = ""


            # Create a list of models to evaluate
            models = [
            ('Random Forest', RandomForestClassifier(random_state=42),
                {'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]}),  # Add hyperparameters for Random Forest
            ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
                {'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for Gradient Boosting
            ('Support Vector Machine', SVC(random_state=42, class_weight='balanced'),
                {'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']}),  # Add hyperparameters for SVM
            # ('Logistic Regression', LogisticRegression(random_state=42, class_weight='balanced'),
            #     {'C': [0.1, 1, 10],
            #     'penalty': ['l1', 'l2']}),  # Add hyperparameters for Logistic Regression
            ('K-Nearest Neighbors', KNeighborsClassifier(),
                {'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']}),  # Add hyperparameters for KNN
            ('Decision Tree', DecisionTreeClassifier(random_state=42),
                {'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]}),  # Add hyperparameters for Decision Tree
            ('Ada Boost', AdaBoostClassifier(random_state=42),
                {'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for Ada Boost
            ('XG Boost', XGBClassifier(random_state=42),
                {'max_depth': randint(3, 6), 
                'learning_rate': uniform(0.01, 0.2),  
                'n_estimators': randint(100, 300),  
                'subsample': uniform(0.8, 0.2)}),  # Add hyperparameters for XG Boost
            ('Naive Bayes', GaussianNB(), {})  # No hyperparameters for Naive Bayes
           ]

            best_model = None
            best_accuracy = 0.0

          # Iterate over the models and evaluate their performance
            for name, model, param_grid in models:
              # Create a pipeline for each model
              pipeline = Pipeline([
                  ('model', model)
              ])

              # Hyperparameter tuning using RandomizedSearchCV for XG Boost
              if name == 'XG Boost':
                  random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                                    n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1)
                  random_search.fit(X_train, y_train)
                  pipeline = random_search.best_estimator_
              # Hyperparameter tuning using GridSearchCV for other models
              elif param_grid:
                  grid_search = GridSearchCV(model, param_grid, cv=2, verbose=0)
                  grid_search.fit(X_train, y_train)
                  pipeline = grid_search.best_estimator_

              # Fit the pipeline on the training data
              pipeline.fit(X_train, y_train)

              # Make predictions on the test data
              y_pred = pipeline.predict(X_test)

              # Calculate accuracy score
              accuracy = accuracy_score(y_test, y_pred)

              # Append model name and accuracy to the list
              model_scores.append({'Model': name, 'Accuracy': accuracy})

              # Convert the list to a DataFrame
              scores_df = pd.DataFrame(model_scores)

              # Print the performance metrics
              print("Model:", name)
              print("Test Accuracy:", accuracy.round(3),"%")
              print()

              # Check if the current model has the best accuracy
              if accuracy > best_accuracy:
                  best_accuracy = accuracy
                  best_model = pipeline
                  best_model_name = name

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return (best_model_name,accuracy)

        except Exception as e:
          raise CustomException(e, sys)
      
      
      