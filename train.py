from src.pipelines.training_pipeline import train_pipeline
from src.logger import logging

if __name__ == '__main__':
    obj = train_pipeline()
    train_data_path, test_data_path = obj.ingestion()
    logging.info("ingesstion complete")
    train_arr, test_arr= obj.transformation(train_data_path, test_data_path)
    logging.info("transformation complete")
    obj.model(train_arr, test_arr)