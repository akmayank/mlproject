import os
import sys
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.components.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training and Testing array")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )
            models = {
                "RandomForest": RandomForestRegressor(),
                # "DecisionTree": DecisionTreeRegressor(),
                # "LinearRegression": LinearRegression(),
                # "GradientBoosting": GradientBoostingRegressor(),
                # "KNN": KNeighborsRegressor(),
                # "XGB": XGBRegressor(),
                # "CatboostRegressor": CatBoostRegressor(),
                # "AdaboostRegressor": AdaBoostRegressor()
            }
            logging.info("Getting All The models")
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test,
                                               models = models)
            logging.info("Generated Model Reports")
            df = pd.DataFrame(model_report, columns=["model_name", "model_score"])
            df.sort_values(["model_score"], ascending=False, inplace=True)
            df.reset_index(inplace=True, drop=True)
            best_model_score = df["model_score"].iloc[0]
            best_model_name = df["model_name"].iloc[0]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset.")
            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predict = best_model.predict(X_test)
            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)