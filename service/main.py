from fastapi import FastAPI
from typing import List, Dict
from joblib import load

app = FastAPI()

@app.post(" /prediction")
def get_predictions(feature_vector : List[float], score : bool = False) -> Dict:
    response = {}
    clf = load('../model.joblib')
    response['is_inlier'] = clf.predict(feature_vector)
    if score:
        response['anomaly_score'] = clf.score_samples(feature_vector)
    return response

@app.get(" /model_information")
def get_hyperparams() -> Dict:
    clf = load('../model.joblib')
    return clf.get_params()