from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/model.pkl")

app = FastAPI(
    title="Iris Flower Prediction API",
    description="REST API for predicting Iris flower species",
    version="1.0"
)


# Input data format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {
        "message": "Iris Prediction API is running"
    }


@app.post("/predict")
def predict(data: IrisInput):

    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(input_data)[0]

    species = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    return {
        "prediction": int(prediction),
        "species": species[int(prediction)]
    }