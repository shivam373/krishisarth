from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load models
crop_model = joblib.load("model/final_model_rf.pkl")
crop_le = joblib.load("model/label_encoder.pkl")

fertility_model = joblib.load("model/final_pipeline.pkl")

# -----------------------------
# Input Schemas
# -----------------------------

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class FertilityInput(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def home():
    return {"message": "Multi-Model API Running 🚀"}


@app.post("/predict-crop")
def predict_crop(data: CropInput):
    columns = ['N','P','K','temperature','humidity','ph','rainfall']

    input_data = pd.DataFrame([[ 
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]], columns=columns)

    pred = crop_model.predict(input_data)
    crop = crop_le.inverse_transform(pred)[0]

    return {"crop_prediction": crop}


@app.post("/predict-fertility")
def predict_fertility(data: FertilityInput):
    columns = ['N','P','K','pH','EC','OC','S','Zn','Fe','Cu','Mn','B']

    input_data = pd.DataFrame([[ 
        data.N, data.P, data.K,
        data.pH, data.EC, data.OC,
        data.S, data.Zn, data.Fe,
        data.Cu, data.Mn, data.B
    ]], columns=columns)

    pred = fertility_model.predict(input_data)
    
    if int(pred[0]) == 0:
        return {"fertility_prediction": "Low"}
    elif pred[0] == 1:
        return {"fertility_prediction": "Medium"}
    else:
        return {"fertility_prediction": "High"}

    # return {"fertility_prediction": int(pred[0])}