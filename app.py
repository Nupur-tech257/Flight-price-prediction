from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from flightpriceprediction.pipeline.predict import Customdata,Predictpipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=Customdata(
            airline=request.form.get('airline'),
            flight=request.form.get('flight'),
            source_city=request.form.get('source_city'),
            departure_time=request.form.get('departure_time'),
            stops=request.form.get('stops'),
            arrival_time=request.form.get('arrival_time'),
            destination_city=request.form.get('destination_city'),
            Class=request.form.get('Class'),
            duration=float(request.form.get('duration')),
            days_left=float(request.form.get('days_left'))

        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=Predictpipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=round(results[0]))
    

if __name__=="__main__":
    app.run(host="0.0.0.0") 