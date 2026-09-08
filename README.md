# Deploying a Pre-Trained Machine Learning Model using FastAPI and Docker

## Objective

The objective of this project is to deploy a pre-trained Scikit-learn machine learning model as a REST API using FastAPI and containerize the application using Docker.

## Technologies Used

- Python
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Docker
- REST API
- cURL / Postman

## Machine Learning Model

The Iris dataset provided by Scikit-learn is used for this project.

The model uses:

- StandardScaler for preprocessing
- Logistic Regression for classification

The trained model achieved an accuracy of approximately 93.33%.

The trained model is saved as:

```text
models/model.pkl