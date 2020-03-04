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

DTC_model = joblib.load(MODEL_FILE)

# Prediction
import numpy as np
mock_unity_data_to_predict = [6.5,3.0,5.5,1.8] # Typical array of floats

array_dimention = len(mock_unity_data_to_predict)
prepared_unity_data = np.ndarray((array_dimention,), buffer=np.array(mock_unity_data_to_predict), dtype=float) # Special ndarray type
X_normalized = preprocessing.normalize([prepared_unity_data], norm='l2')
DTC_prediction = DTC_model.predict(X_normalized)

print(DTC_prediction)
