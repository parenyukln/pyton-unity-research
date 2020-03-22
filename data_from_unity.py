# Сбор данных с Unity
import pandas as pd  
from numpy import median
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
import config

# Собрынные данные из Unity (сырые: просто история)
data = pd.read_csv(config.UNITYDATA_FILE) 

# Обработка данных, отсеивание "плохих" записей
target_param_array = data[config.TARGET_PARAM_NAME]
target_param_array_median = median(target_param_array)
filtered_data = data.loc[data[config.TARGET_PARAM_NAME] >= target_param_array_median]

# Выгружаем csv файл
filtered_data.to_csv(path_or_buf=config.SCRIPTOUTPUT_FILE, index=False)

