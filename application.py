from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.preprocessing import MinMaxScaler
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
application=Flask(__name__)

app = application

##Route for home page

@app.route('/')
def index():
  return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
  if request.method=='GET':
    return render_template('home.html')
  else:
    data = CustomData(
      gender = request.form.get('gender'),
      SeniorCitizen = request.form.get('SeniorCitizen'),
      Partner = request.form.get('Partner'),
      Dependents = request.form.get('Dependents'),
      tenure = request.form.get('tenure') ,
      PhoneService = request.form.get('PhoneService'),
      MultipleLines = request.form.get('MultipleLines'),
      InternetService = request.form.get('InternetService'),
      OnlineSecurity = request.form.get('OnlineSecurity'),
      OnlineBackup=request.form.get('OnlineBackup'),
      DeviceProtection = request.form.get('DeviceProtection'),
      TechSupport = request.form.get('TechSupport'),
      StreamingTV = request.form.get('StreamingTV'),
      StreamingMovies =  request.form.get('StreamingMovies'),
      Contract = request.form.get('Contract'),
      PaperlessBilling = request.form.get('PaperlessBilling'),
      PaymentMethod = request.form.get('PaymentMethod'),
      MonthlyCharges = request.form.get('MonthlyCharges'),
      TotalCharges = request.form.get('TotalCharges'),
    )
    
    pred_df = data.get_data_as_data_frame()    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html',results=results[0])
    
    
    
if __name__=="__main__":
  app.run(host="0.0.0.0",debug=True)