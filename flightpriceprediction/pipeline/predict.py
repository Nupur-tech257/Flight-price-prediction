import os ,sys
from flightpriceprediction.logger import logging
from flightpriceprediction.exception import CustomException
from dataclasses import dataclass
from flightpriceprediction.utils import load_object
import pandas as pd

@dataclass
class Predictpipeline:
    def __init__(self):
        pass
    
    def predict(self,data):
        try:
            model_filepath=os.path.join("artifact","model.pkl")
            preprocessor_filepath=os.path.join("artifact","preprocessor.pkl")
            preprocessor=load_object(preprocessor_filepath)
            model=load_object(model_filepath)
            data=preprocessor.transform(data)
            result=model.predict(data)
            return result
        except  Exception as e:
            raise CustomException(e,sys)
        
class Customdata:
    def __init__(self,airline:str,flight:str ,source_city:str,departure_time:str,stops:str ,arrival_time:str,destination_city:str,Class :str,duration:float,days_left:int):
        self.airline=airline
        self.flight=flight
        self.source_city=source_city
        self.departure_time=departure_time
        self.stops=stops
        self.arrival_time=arrival_time
        self.destination_city=destination_city
        self.Class=Class
        self.duration=duration
        self.days_left=days_left

    def get_data_as_dataframe(self):
        try:
            custom_data={
            'airline':[self.airline],
            'flight':[self.flight],
            'source_city':[self.source_city],
            'departure_time':[self.departure_time],
            'stops':[self.stops],
            'arrival_time':[self.arrival_time],
            'destination_city':[self.destination_city],
            'class':[self.Class],
            'duration':[self.duration],
            'days_left':[self.days_left],
            }
            return pd.DataFrame(custom_data)
        except  Exception as e:
            raise CustomException(e,sys)
        
