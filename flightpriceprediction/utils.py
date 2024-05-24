import os,sys
import pandas as pd
import numpy as np
import pickle
from flightpriceprediction.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
        
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        mse_report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            mse=mean_squared_error(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            mse_report[list(models.keys())[i]]= mse

        return report,mse_report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def outlier(df):
    a='price'
    print(df[a].describe().values)
    iqr=df[a].describe().values[6]-df[a].describe().values[4]
    print(iqr)
    lowerfence=df[a].describe().values[4]-(1.5*iqr)
    upperfence=df[a].describe().values[6]+(1.5*iqr)
    print(lowerfence)
    print(upperfence)
    for i in range(1,2000):
        if df[a].iloc[i]<lowerfence:
            df[a].iloc[i]=lowerfence
        if df[a].iloc[i]>upperfence:
            df[a].iloc[i]=upperfence