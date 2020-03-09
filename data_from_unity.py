# Сбор данных с Unity
import pandas as pd  
from numpy import median
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

UNITYDATA_FILE = 'python-ml/unity_data.csv'
SCRIPTOUTPUT_FILE = 'python-ml/prepared_data.csv'

# Собрынные данные из Unity (сырые: просто история)
data = pd.read_csv(UNITYDATA_FILE) 
target_param_name = 'Health' # Создать отдельный конфигурационный файл с помощью unity-python

# Обработка данных, отсеивание "плохих" записей
# TODO: Проанализировать границу для фильтра целевого параметра и отсеять записи, где целевой параметр ниже границы
# Варианты: Ср. арифметическое, ср. квадратическое, !медиана!, мода
target_param_array = data[target_param_name]
target_param_array_median = median(target_param_array)
filtered_data = data.loc[data[target_param_name] >= target_param_array_median]

# Выгружаем csv файл
filtered_data.to_csv(path_or_buf=SCRIPTOUTPUT_FILE, index=False)

