import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd


# from data_ingestion import DataIngestion
# from data_transformation import DataTransformation
# from model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class train_pipeline:
    def ingestion(self):
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        return train_data_path, test_data_path
    
    def transformation(self,train_data_path, test_data_path):
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        return train_arr,test_arr

    def model(self,train_arr, test_arr):
        model_trainer = ModelTrainer()
        model_trainer.initate_model_training(train_arr, test_arr)



# if __name__ == '__main__':
# obj = DataIngestion()
# train_data_path, test_data_path = obj.initiate_data_ingestion()
# print(train_data_path, test_data_path)

# data_transformation = DataTransformation()

# train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
#     train_data_path, test_data_path)

# model_trainer = ModelTrainer()
# model_trainer.initate_model_training(train_arr, test_arr)
