import os,sys
import pandas as pd
import numpy as np
from flightpriceprediction.logger import logging
from flightpriceprediction.exception import CustomException
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from flightpriceprediction.utils import save_object,outlier

@dataclass
class DataTransformationconfig :
    preprocessor_filepath=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_datatransformation_object(self):
        '''this data is responsible for data transformation'''
        try:
            logging.info("data transformation has started")
            numerical_data=['duration', 'days_left']
            categorical_data=['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

            num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                        ("scaler",StandardScaler())])
            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                          ('one hot encoder',OneHotEncoder(handle_unknown='infrequent_if_exist')),
                                          ('scaler',StandardScaler(with_mean=False))
                                        ])
            preprocessor=ColumnTransformer([("numerical pipeline",num_pipeline,numerical_data),
                                            ("categorical pipeline",cat_pipeline,categorical_data)])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
            logging.info("data transformation started")
            try:
                train_df=pd.read_csv(train_path)
                train_df.replace('-','',regex=True)
                train_df.replace('_','',regex=True)
                train_df.replace(' ','',regex=True)
                test_df=pd.read_csv(test_path)
                test_df.replace('-','',regex=True)
                test_df.replace('_','',regex=True)
                test_df.replace(' ','',regex=True)
                #outlier(train_df)
                #outlier(test_df)
                preprocessor_obj=self.get_datatransformation_object()
                target_column_name='price'
                train_df_X=train_df.drop(columns=[target_column_name],axis=1)
                train_df_Y=train_df[target_column_name]

                test_df_X=test_df.drop(columns=[target_column_name],axis=1)
                test_df_Y=test_df[target_column_name]

                logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
        
                train_arr_X=preprocessor_obj.fit_transform(train_df_X)
                test_arr_X=preprocessor_obj.transform(test_df_X)

                #train_arr = np.concatenate([np.array(train_arr_X),np.array(train_df_Y)],axis=1)
                #test_arr = np.c_[test_arr_X, np.array(test_df_Y)]

                logging.info(f"Saved preprocessing object.")

                save_object(file_path=self.data_transformation_config.preprocessor_filepath,
                    obj=preprocessor_obj)

                return (train_arr_X,test_arr_X,train_df_Y,test_df_Y,self.data_transformation_config.preprocessor_filepath)
            except Exception as e:
                raise CustomException(e,sys)



