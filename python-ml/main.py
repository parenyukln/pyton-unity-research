import pandas as pd  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

data = pd.read_csv('Iris.csv')
data.drop('Id', axis=1, inplace=True)  

# ".iloc" принимает row_indexer, column_indexer  
X = data.iloc[:,:-1].values  
# Параметр, который классифицируем
y = data['Species'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# Модель
DTC_model = DecisionTreeClassifier(random_state=0)
DTC_model.fit(X_train, y_train) 

DTC_prediction = DTC_model.predict(X_test)

print(DTC_prediction)
# Оценка точности — простейший вариант оценки работы классификатора
print(accuracy_score(DTC_prediction, y_test))  
# Но матрица неточности и отчёт о классификации дадут больше информации о производительности
print(confusion_matrix(DTC_prediction, y_test))  