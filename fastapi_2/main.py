import dill

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
app = FastAPI()

with open('rfr_v1.pkl', 'rb') as f: 
    reloaded_model = dill.load(f) 
    
class Payload(BaseModel): 
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.post("/")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    y_hat = reloaded_model.predict(df)
    return {"prediction": y_hat[0]}
