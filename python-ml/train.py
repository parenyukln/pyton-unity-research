import pandas as pd  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

INPUT_PREPARED_DATA_FILE = 'prepared_data.csv'
MODEL_FILE = 'model.pkl'

# Данные из Unity после обработки (подготовленные)
data = pd.read_csv(INPUT_PREPARED_DATA_FILE)
data.drop('Id', axis=1, inplace=True)

# ".iloc" принимает row_indexer, column_indexer  
X = data.iloc[:,:-1].values  
# Параметр, который классифицируем
y = data['Action'] 

# Normalize
X_normalized = preprocessing.normalize(X, norm='l2')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.01, random_state=27)

# Модель
DTC_model = DecisionTreeClassifier(random_state=0)

# Обучение модели
DTC_model.fit(X_train, y_train) 

# Выгрузка обученной модели 
import joblib

joblib.dump(DTC_model, MODEL_FILE)

# Оценка точности — простейший вариант оценки работы классификатора
#print(accuracy_score(DTC_prediction, y_test))  
# Но матрица неточности и отчёт о классификации дадут больше информации о производительности
#print(confusion_matrix(DTC_prediction, y_test))  

