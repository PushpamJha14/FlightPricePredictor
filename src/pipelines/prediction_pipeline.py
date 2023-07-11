import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            # Date of Journey
            features["Journey_day"] = features['Date_of_Journey'].str.split('-').str[0].astype(int)
            features["Journey_month"] = features['Date_of_Journey'].str.split('-').str[1].astype(int)
            features["Journey_year"] = features['Date_of_Journey'].str.split('-').str[2].astype(int)
            features.drop(["Date_of_Journey"], axis = 1, inplace = True)

            # Dep_Time
            features["Dep_hour"] = pd.to_datetime(features["Dep_Time"]).dt.hour
            features["Dep_min"] = pd.to_datetime(features["Dep_Time"]).dt.minute
            features.drop(["Dep_Time"], axis = 1, inplace = True)

            # Arrival_Time
            features["Arrival_hour"] = pd.to_datetime(features.Arrival_Time).dt.hour
            features["Arrival_min"] = pd.to_datetime(features.Arrival_Time).dt.minute
            features.drop(["Arrival_Time"], axis = 1, inplace = True)

            features['Duration'] = (features['Arrival_hour']*60+features['Arrival_min']) - (features['Dep_hour']*60+features['Dep_min'])

            # Encoding Total_Stops for Training and Testing data
            features['Total_Stops']=features['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})
            

            # Encoding Additional Info of Training and Testing Data
            features['Additional_Info']=features['Additional_Info'].map({ 'No info':4, 'In-flight meal not included':2, 'No check-in baggage included':1, '1 Short layover':3,
                                                                'No Info':1, '1 Long layover':3, 'Change airports':3, 'Business class':5, 'Red-eye flight':2,
                                                                '2 Long layover':3 })

            # Encoding Airline of Training and Testing Data
            features['Airline']=features['Airline'].map({ 'IndiGo':1, 'Air India':2, 'Jet Airways':2, 'SpiceJet':1, 'Multiple carriers':2, 'GoAir':1, 'Vistara':3,
                                                        'Air Asia':1, 'Vistara Premium economy':3, 'Jet Airways Business':5, 'Multiple carriers Premium economy':4,
                                                        'Trujet':1 })

            # One Hot Encoding
            encoder=OneHotEncoder()
            df=pd.DataFrame(encoder.fit_transform(features[['Source','Destination']]).toarray(),columns=encoder.get_feature_names_out())
            features=pd.concat([features,df],axis=1)
            features.drop(['Source','Destination'], axis=1,inplace=True)
            logging.info(features.columns)

            if 'Source_Banglore' not in features.columns:
                features['Source_Banglore'] = 0
            if 'Source_Chennai' not in features.columns:
                features['Source_Chennai'] = 0
            if 'Source_Delhi' not in features.columns:
                features['Source_Delhi'] = 0
            if 'Source_Kolkata' not in features.columns:
                features['Source_Kolkata'] = 0
            if 'Source_Mumbai' not in features.columns:
                features['Source_Mumbai'] = 0
            if 'Destination_Banglore' not in features.columns:
                features['Destination_Banglore'] = 0
            if 'Destination_Cochin' not in features.columns:
                features['Destination_Cochin'] = 0
            if 'Destination_Delhi' not in features.columns:
                features['Destination_Delhi'] = 0
            if 'Destination_Hyderabad' not in features.columns:
                features['Destination_Hyderabad'] = 0
            if 'Destination_Kolkata' not in features.columns:
                features['Destination_Kolkata'] = 0
            if 'Destination_New Delhi' not in features.columns:
                features['Destination_New Delhi'] = 0
            logging.info(features.columns)
            features = features[['Airline', 'Duration', 'Total_Stops', 'Additional_Info',
       'Journey_day', 'Journey_month', 'Journey_year', 'Dep_hour', 'Dep_min',
       'Arrival_hour', 'Arrival_min', 'Source_Banglore', 'Source_Chennai',
       'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi']]
            
            logging.info(features[['Duration','Arrival_hour','Arrival_min','Dep_hour','Dep_min']])
            logging.info(features['Arrival_hour']*60+features['Arrival_min'])
            logging.info(features['Dep_hour']*60+features['Dep_min'])

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Airline:str,
                 Source:str,
                 Destination:str,
                 Total_Stops:str,
                 Date_of_Journey:str,
                 Arrival_Time:str,
                 Dep_Time:str,
                 Additional_Info:str
                 ):
        
        self.Airline=Airline
        self.Source=Source
        self.Destination=Destination
        self.Total_Stops=Total_Stops
        self.Date_of_Journey=Date_of_Journey
        self.Arrival_Time=Arrival_Time
        self.Dep_Time = Dep_Time
        self.Additional_Info = Additional_Info

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Airline':[self.Airline],
                'Source':[self.Source],
                'Destination':[self.Destination],
                'Total_Stops':[self.Total_Stops],
                'Date_of_Journey':[self.Date_of_Journey],
                'Arrival_Time':[self.Arrival_Time],
                'Dep_Time':[self.Dep_Time],
                'Additional_Info':[self.Additional_Info]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


