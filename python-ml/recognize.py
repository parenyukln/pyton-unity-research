# Сбор данных с Unity
import pandas as pd  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

MODEL_FILE = 'model.pkl'

# Загрузка обученной модели 
import joblib
import numpy as np

DTC_model = joblib.load(MODEL_FILE)

# Prediction
class RecognizeModel():
    def __init__(self, data_from_unity):
        self.data_to_recognize = data_from_unity

    def recognize(self):
        array_dimention = len(self.data_to_recognize)
        prepared_unity_data = np.ndarray((array_dimention,), buffer=np.array(self.data_to_recognize), dtype=float) # Special ndarray type
        X_normalized = preprocessing.normalize([prepared_unity_data], norm='l2')
        DTC_prediction = DTC_model.predict(X_normalized)

        return DTC_prediction
