# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# File system management
import os

import joblib
from pydantic import create_model
from fastapi import FastAPI

# https://realpython.com/fastapi-python-web-apis/
# launch API in local
# cd C:\Users\disch\Documents\OpenClassrooms\Workspace\20220411-Projet_7_Implementez_un_modele_de_scoring\Projet_7
# uvicorn P7_API:app --reload

# Basic URL
# http://127.0.0.1:8000

# Documentation
# http://127.0.0.1:8000/docs

if not os.path.exists('bin'):
    print('problem, bin folder not existing in path')
else:

    #
    # Deserializing
    #

    ohe = joblib.load('bin/ohe.joblib')
    scaler = joblib.load('bin/std_scaler.joblib')
    model = joblib.load('bin/model.joblib')
    
    #
    # Building Model
    #
    
    data_model_dict = joblib.load('bin/data_dict.joblib')
    ClientModel = create_model("ClientModel", **data_model_dict)
        
    categorical_feats, numeric_feats = [], []
    for key in data_model_dict:
        if data_model_dict[key][0] == str:
            categorical_feats.append(key)
        else:
            numeric_feats.append(key) 
    print('\nCategorical features: \t', categorical_feats)
    print('\nNumeric features: \t', numeric_feats)

    #
    # API
    #

    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Loan repayment API deployed"}
    
    @app.get("/model")
    async def get_model():
        return model

    @app.post("/loan_repayment/")
    async def predict_loan_repayment(client: ClientModel):
        print('Client: \t', client)
        
        # Feature engineering
        client.CREDIT_INCOME_PERCENT = client.AMT_CREDIT / client.AMT_INCOME_TOTAL
        client.ANNUITY_INCOME_PERCENT = client.AMT_ANNUITY / client.AMT_INCOME_TOTAL
        client.CREDIT_TERM = client.AMT_ANNUITY / client.AMT_CREDIT
        client.DAYS_EMPLOYED_PERCENT = client.DAYS_EMPLOYED / client.DAYS_BIRTH
        print('\nFeatures shape after feature engineering: \t', client)
    
        # ohe
        cat_array = pd.DataFrame(data=[[getattr(client, categorical_feat) for categorical_feat in categorical_feats]], columns=categorical_feats)
        # cat_array = client[categorical_feats_filtered]
        print('\nFeatures shape before one-hot encoding: \t', cat_array.shape)
        # one-hot encoding of categorical variables
        cat_array_encoded = ohe.transform(cat_array).toarray()
        print('Features shape after one-hot encoding: \t\t', cat_array_encoded.shape)
    
        # scaler
        num_array = pd.DataFrame(data=[[getattr(client, numeric_feat) for numeric_feat in numeric_feats]], columns=numeric_feats)
        # # num_array = client[numeric_feats]
        print('\nFeatures shape before imputing/scaling: \t', num_array.shape)
        num_array = scaler.transform(num_array)
        print('Features shape after imputing/scaling: \t\t', num_array.shape)
    
        # predict
        X = np.concatenate([cat_array_encoded, num_array], axis=1)
        X = np.asarray(X)    
        print('X to predict: \t\t', X)
        y_proba = model.predict_proba(X)[0, 1]
        print('Target = 1 probability: \t\t', y_proba)
    
        return y_proba


