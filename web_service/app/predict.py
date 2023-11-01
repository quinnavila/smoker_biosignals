
import joblib
import numpy as np
import os
import pandas as pd
from typing import Dict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier





app = FastAPI()


categorical_features = ['hearing(left)', 'hearing(right)', 'dental caries']
numerical_features = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
                      'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar', 
                      'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 
                      'Urine protein', 'serum creatinine', 'AST', 'ALT', 'Gtp', 'Log_HDL_LDL_Ratio']

def load_model(model_filename):
    """Load model from a file.

    This function loads model from the specified file.

    Args:
        filename (str): The filename of the XGBoost model.

    Returns:
        model
    """
    
    # Get the directory of the currently executing app
    app_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the model file
    model_path = os.path.join(app_directory, model_filename)
    model = joblib.load(model_path)

    return model

def prepare_features(bio_signals):
    """
    Prepare the input features for prediction 

    Parameters:
        bio_signals (Dict[str, float]): A dictionary containing bio_signals information.

    Returns:
        pd.DataFrame: A DataFrame with the vital information.
    """
    df = pd.DataFrame([bio_signals])
    df['Log_HDL_LDL_Ratio'] = np.log1p(df['HDL'] / df['LDL'])
    
    return df

def predict(X):
    """
    Make a prediction using the input features.

    Parameters:
        features (pd.DataFrame): DataFrame containing the input features.

    Returns:
        int: The predicted outcome (0 or 1).
    """
    model = load_model("random_forest_model.pkl")

    preprocessor = model.named_steps['preprocessor']
    
    preprocessed_data = preprocessor.transform(X)

    # Make predictions
    preds_binary = model.named_steps['classifier'].predict(preprocessed_data)
    print(preds_binary)
    return int(preds_binary[0])

@app.post("/predict", status_code=200)
def predict_endpoint(bio_signals: Dict[str, float]):
    """
    Endpoint to make a prediction based on biosignals.

    Parameters:
        vitals (Dict[str, float]): A dictionary containing biosignals.

    Returns:
        JSONResponse: A JSON response containing the predicted outcome.
    """
    try:
        
        features = prepare_features(bio_signals)
        
        pred = predict(features)

        result = {
            'outcome': pred
        }

        return JSONResponse(content=result)

    except Exception as e:
        # Handle any errors that might occur during processing
        error_msg = {'predict_endpoint error': str(e)}
        return JSONResponse(content=error_msg, status_code=500)

@app.get("/healthcheck")  # Route for ELB health checks
def healthcheck():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)