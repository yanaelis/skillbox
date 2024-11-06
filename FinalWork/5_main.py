import dill as pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('model/cars_pipe.pkl', 'rb') as file:
   model = pickle.load(file)


class Prediction(BaseModel):
    session_id: str
    Result: int


class Form(BaseModel):
    session_id: str
    client_id: float |None
    visit_date: str| None
    visit_time: str| None
    visit_number: int| None
    utm_source: str| None
    utm_medium: str| None
    utm_adcontent: str | None
    utm_campaign: str | None
    utm_keyword: str| None
    device_category: str| None
    device_os: str| None
    device_brand: str| None
    device_model: str| None
    device_screen_resolution: str| None
    device_browser: str| None
    geo_country: str| None
    geo_city: str| None


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame([form.model_dump()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'Result': y[0]
    }



