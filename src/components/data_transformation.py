from sklearn.impute import SimpleImputer          # Handling Missing Values
from sklearn.preprocessing import StandardScaler  # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder  # Ordinal Encoding
from sklearn.preprocessing import OneHotEncoder   # One Hot Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


## Data Ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_cols = ['Airline', 'Duration', 'Total_Stops', 'Additional_Info',
                                'Journey_day', 'Journey_month', 'Journey_year', 'Dep_hour', 'Dep_min',
                                'Arrival_hour', 'Arrival_min', 'Source_Banglore', 'Source_Chennai',
                                'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                                'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
                                'Destination_Hyderabad', 'Destination_Kolkata',
                                'Destination_New Delhi']
            categorical_cols = []
    

            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
                ]

            )
            logging.info('a')
            preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols),
            ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            logging.info('b')
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:

            logging.info("Error in Data Trnasformation")
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            logging.info('c')
            # Dropping NULL values
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            # Duration convert hours in minutes
            train_df['Duration'] = train_df['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
            test_df['Duration'] = test_df['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

            # FOR TRAINING DATA
            # Date_of_Journey
            train_df["Journey_day"] = train_df['Date_of_Journey'].str.split('/').str[0].astype(int)
            train_df["Journey_month"] = train_df['Date_of_Journey'].str.split('/').str[1].astype(int)
            train_df["Journey_year"] = train_df['Date_of_Journey'].str.split('/').str[2].astype(int)
            train_df.drop(["Date_of_Journey"], axis = 1, inplace = True)

            # Dep_Time
            train_df["Dep_hour"] = pd.to_datetime(train_df["Dep_Time"]).dt.hour
            train_df["Dep_min"] = pd.to_datetime(train_df["Dep_Time"]).dt.minute
            train_df.drop(["Dep_Time"], axis = 1, inplace = True)

            # Arrival_Time
            train_df["Arrival_hour"] = pd.to_datetime(train_df.Arrival_Time).dt.hour
            train_df["Arrival_min"] = pd.to_datetime(train_df.Arrival_Time).dt.minute
            train_df.drop(["Arrival_Time"], axis = 1, inplace = True)


            # FOR TESTING DATA
            # Date_of_Journey
            test_df["Journey_day"] = test_df['Date_of_Journey'].str.split('/').str[0].astype(int)
            test_df["Journey_month"] = test_df['Date_of_Journey'].str.split('/').str[1].astype(int)
            test_df["Journey_year"] = test_df['Date_of_Journey'].str.split('/').str[2].astype(int)
            test_df.drop(["Date_of_Journey"], axis = 1, inplace = True)

            # Dep_Time
            test_df["Dep_hour"] = pd.to_datetime(test_df["Dep_Time"]).dt.hour
            test_df["Dep_min"] = pd.to_datetime(test_df["Dep_Time"]).dt.minute
            test_df.drop(["Dep_Time"], axis = 1, inplace = True)

            # Arrival_Time
            test_df["Arrival_hour"] = pd.to_datetime(test_df.Arrival_Time).dt.hour
            test_df["Arrival_min"] = pd.to_datetime(test_df.Arrival_Time).dt.minute
            test_df.drop(["Arrival_Time"], axis = 1, inplace = True)

            # Encoding Total_Stops for Training and Testing data
            train_df['Total_Stops']=train_df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})
            test_df['Total_Stops']=test_df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})

            # Encoding Additional Info of Training and Testing Data
            train_df['Additional_Info']=train_df['Additional_Info'].map({ 'No info':4, 'In-flight meal not included':2, 'No check-in baggage included':1, '1 Short layover':3,
                                                                'No Info':1, '1 Long layover':3, 'Change airports':3, 'Business class':5, 'Red-eye flight':2,
                                                                '2 Long layover':3 })
            test_df['Additional_Info']=test_df['Additional_Info'].map({ 'No info':4, 'In-flight meal not included':2, 'No check-in baggage included':1, '1 Short layover':3,
                                                                'No Info':1, '1 Long layover':3, 'Change airports':3, 'Business class':5, 'Red-eye flight':2,
                                                                '2 Long layover':3 })

            # Encoding Airline of Training and Testing Data
            train_df['Airline']=train_df['Airline'].map({ 'IndiGo':1, 'Air India':2, 'Jet Airways':2, 'SpiceJet':1, 'Multiple carriers':2, 'GoAir':1, 'Vistara':3,
                                                        'Air Asia':1, 'Vistara Premium economy':3, 'Jet Airways Business':5, 'Multiple carriers Premium economy':4,
                                                        'Trujet':1 })
            test_df['Airline']=test_df['Airline'].map({ 'IndiGo':1, 'Air India':2, 'Jet Airways':2, 'SpiceJet':1, 'Multiple carriers':2, 'GoAir':1, 'Vistara':3,
                                                    'Air Asia':1, 'Vistara Premium economy':3, 'Jet Airways Business':5, 'Multiple carriers Premium economy':4,
                                                    'Trujet':1 })

            # One Hot Encoding
            encoder=OneHotEncoder()
            df=pd.DataFrame(encoder.fit_transform(train_df[['Source','Destination']]).toarray(),columns=encoder.get_feature_names_out())
            train_df=pd.concat([train_df,df],axis=1)
            train_df.drop(['Source','Destination','Route'], axis=1,inplace=True)

            df=pd.DataFrame(encoder.fit_transform(test_df[['Source','Destination']]).toarray(),columns=encoder.get_feature_names_out())
            test_df=pd.concat([test_df,df],axis=1)
            test_df.drop(['Source','Destination','Route'], axis=1,inplace=True)


            target_column_name = 'Price'
            drop_columns = [target_column_name]

            ## features into independent and dependent features
            train_df.dropna(inplace=True)
            logging.info('e')
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]


            ## apply the transformation
            logging.info(input_feature_test_df.columns)
            logging.info('f')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info('g')
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info('Processsor pickle in created and saved')

            #######################################################
            # input_feature_train_df.dropna(inplace=True)
            # target_feature_train_df.dropna(inplace=True)
            
            logging.info(train_df.isnull().sum())
            logging.info(test_df.isnull().sum())
            #####################################################



            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e, sys)
