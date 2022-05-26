# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# File system management
import os

import joblib
from pydantic import create_model #, BaseModel
# from typing import List

from fastapi import FastAPI, Response #, Request
from fastapi.responses import FileResponse
import json

# https://realpython.com/fastapi-python-web-apis/
# launch API in local
# cd C:\Users\disch\Documents\OpenClassrooms\Workspace\20220411-Projet_7_Implementez_un_modele_de_scoring\Projet_7\repository\scoring-bank-p7
# uvicorn app:app --reload

# Basic URL
# http://127.0.0.1:8000

# Documentation
# http://127.0.0.1:8000/docs

def readFileToList(filepath, label):
    # opening the file in read mode
    _file = open(filepath, 'r')
    # reading the file
    data = _file.read()
    # replacing end splitting the text 
    # when newline ('\n') is seen.
    data_list = data.split("\n")
    data_list.remove("")
    print("\n" + label, data_list)
    _file.close()
    
    return data_list


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
    # Data
    #
    
    features_for_model_prediction = readFileToList('bin/features_for_model_prediction.txt', 'Features for model prediction:')
    features_for_dashboard_table = readFileToList('bin/features_for_dashboard_table.txt', 'Features for dashboard main table:')
    
    compression_opts = dict(method='zip', archive_name='data.csv')
    db_test = pd.read_csv('bin/data.zip', compression=compression_opts)  
    print('\nData shape: \t', db_test.shape)
    db_test = db_test.reset_index(drop=True)
    
    shap_expected_value = float(readFileToList('bin/shap_expected_value.txt', 'SHAP expected value:')[0])
    
    #shap_shap_values = np.genfromtxt('bin/shap_shap_values.txt', dtype=float, delimiter=" ")
    #print("\nSHAP SHAP value:", shap_shap_values)
    
    print("\n")
    
    #
    # API
    #

    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Loan repayment API deployed"}
    
    #@app.get("/model")
    #async def get_model(response: Response):
    #    response.headers["Accept"] = 'application/octet-stream'
    #    return FileResponse(path='bin/model.joblib', filename='model.joblib', media_type='application/octet-stream') 
    
    @app.get("/features_for_model_prediction")
    async def get_features_for_model_prediction():
        return features_for_model_prediction
    
    @app.get("/features_for_dashboard_table")
    async def get_features_for_dashboard_table():
        return features_for_dashboard_table
    
    @app.get("/clients")
    async def get_all_clients():    
        return Response(db_test.to_json(orient="records"), media_type="application/json")
    
    @app.get("/shap_expected_value")
    async def get_shap_expected_value():
        return shap_expected_value
    
    @app.get("/shap_shap_values")
    async def get_shap_shap_values():    
        #return Response(json.dumps(shap_shap_values.tolist()), media_type="application/json")
        return FileResponse(path='bin/shap_shap_values.npz', filename='shap_shap_values.npz', media_type='application/octet-stream') 
    
    #class Criterion(BaseModel):
    #    feature: str
    #    value: str | int
    #    
    #class Filters(BaseModel):
    #    data: List[Criterion]
    #
    #def filter(df, col, value):
    #	'''Fonction pour filtrer le dataframe selon la colonne et la valeur définies'''
    #	if value != 'All':
    #		db_filtered = df.loc[df[col]==value]
    #	else:
    #		db_filtered = df
    #	return db_filtered
    
    #    @app.post("/filter_clients")
    #     async def filter_clients(filters: Filters):
    #         print('filters: \t', filters)
    #         print('filters.data: \t', filters.data)
            
    #         #Affichage du dataframe selon les filtres définis
    #         db_display=db_test[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
    #         'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
    #         'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
    #         db_display['YEARS_BIRTH']=db_display['YEARS_BIRTH'].astype(str)
    #         db_display['CNT_CHILDREN']=db_display['CNT_CHILDREN'].astype(str)
    #         db_display['AMT_INCOME_TOTAL']=db_test['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
    #         db_display['AMT_CREDIT']=db_test['AMT_CREDIT'].apply(lambda x: int(x))
    #         db_display['AMT_ANNUITY']=db_test['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))
            
    #         for criteria in filters.data:
    #             print(criteria.feature + " - " + criteria.value)
    #             db_display=filter(db_display, criteria.feature, criteria.value)
            
    #         #return {
    #         #    "status" : "SUCCESS",
    #         #    "data" : db_display
    #         #}
        
    #         return Response(db_display.to_json(orient="records"), media_type="application/json")
    
    @app.post("/loan_repayment")
    async def get_loan_repayment_prediction(client: ClientModel):
        print('Client: \t', client)
            
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
        print('\nFeatures shape for scaling: \t\t\t', num_array.shape)
        num_array = scaler.transform(num_array)
    
        # predict with model
        X = np.concatenate([cat_array_encoded, num_array], axis=1)
        X = np.asarray(X)    
        print('\nX to predict:\n', X)
        y_pred = model.predict_proba(X)[0, 1]
        print('\nProbability of [Target = 1]: \t\t', y_pred)
        
        # For features and features names after feature engineering/one-hot-encoding/scaling
        # FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2.
        # Please use get_feature_names_out instead.
        df_cat = pd.DataFrame(data=cat_array_encoded, columns=ohe.get_feature_names_out())
        df_num = pd.DataFrame(data=num_array, columns=numeric_feats)
        # concatenating df_cat and df_num along columns
        df_all = pd.concat([df_cat, df_num], axis=1)

        return json.dumps({
            "y_pred": y_pred,
            "decision": np.where(np.array([y_pred]) > 0.5040000000000002, "Rejeté", "Approuvé")[0],
            "features": df_all.to_json(orient="records"),
            "feature_names": list(df_all.columns)
        })
            


