import os,sys 
from flightpriceprediction.logger import logging
from flightpriceprediction.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class Dataingestionconfig:
    raw_data=os.path.join("artifact","raw.csv")
    training_data=os.path.join("artifact","train.csv")
    testing_data=os.path.join("artifact","test.csv")

class Dataingestion:
    def __init__(self):
        self.data_ingestion_config=Dataingestionconfig()
    
    def dataingestion(self):
        logging.info("data ingestion started")
        try:
            df=pd.read_csv("notebook/data/flight_Dataset.csv")
            logging.info("data collected")
            makedir=os.makedirs(os.path.dirname(self.data_ingestion_config.training_data),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data,index=False,header=True)
            logging.info("train test split initiated")
            train_data,test_data=train_test_split(df,random_state=42,test_size=0.2)
            train_data.to_csv(self.data_ingestion_config.training_data,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.testing_data,index=False,header=True)
            logging.info("data ingestion completed")
            return(
                self.data_ingestion_config.training_data,
                self.data_ingestion_config.testing_data)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=Dataingestion()  
    train_data,test_data=obj.dataingestion()  