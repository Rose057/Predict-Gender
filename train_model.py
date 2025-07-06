# Импорт всех необходимых библиотек
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import ast
from datetime import datetime


# Загрузка данных из файлов csv
train_df = pd.read_csv('train.csv', sep=';')
train_labels_df = pd.read_csv('train_labels.csv', sep=';')
referer_vectors_df = pd.read_csv('referer_vectors.csv', sep=';')
geo_info_df = pd.read_csv('geo_info.csv', sep=';')
print("Файлы загружены")

# Объединение всех таблиц в одну
train_df = train_df.merge(train_labels_df, on='user_id', how='left')
train_df = train_df.merge(referer_vectors_df, on='referer', how='left')
train_df = train_df.merge(geo_info_df, on='geo_id', how='left')
print("Все таблицы объединены")
# Функция для разбора user_agent. Она извлекает browser и os из user_agent
def from_user_agent(x):
    try:
        us_ag_dict = ast.literal_eval(x)
        return us_ag_dict.get('browser', 'unknown'), us_ag_dict.get('os', 'unknown')
    except:
        return 'unknown', 'unknown'

# Разбор user_agent с применением функции from_user_agent и создание новых колонок browser и os
train_df[['browser', 'os']] = train_df['user_agent'].apply(
    lambda x: pd.Series(from_user_agent(x)))

# Функция для извлечения часа
def hour(x):
    try:
        return datetime.fromtimestamp(x).hour
    except:
        return -1

# Извлечение часа и создание новой колонки hour
train_df['hour'] = train_df['request_ts'].apply(hour)

# Функция для извлечения дня недели
def weekday(x):
    try:
        return datetime.fromtimestamp(x).weekday()
    except:
        return -1

# Извлечение дня недели и создание новой колонки day_of_week
train_df['day_of_week'] = train_df['request_ts'].apply(weekday)

# Функция для извлечения наиболее частого значения
def safe_mode(x):
    try:
        return x.mode()[0]
    except:
        return None

# Заполнение пустых полей в region_id
train_df['region_id'] = train_df['region_id'].fillna('unknown')
print("Преобразование фичей завершено. Далее начинается агрегация по пользователям. Займет чуть больше 5 минут")

# Группировка данных по user_id и вычисление всех признаков
user_features = train_df.groupby('user_id').agg({
    'component0': 'mean', # Средние значения десяти компонент URL
    'component1': 'mean',
    'component2': 'mean',
    'component3': 'mean',
    'component4': 'mean',
    'component5': 'mean',
    'component6': 'mean',
    'component7': 'mean',
    'component8': 'mean',
    'component9': 'mean',
    'request_ts': 'count', # Подсчет количества запросов пользователя
    'referer': lambda x: safe_mode(x), # Самый частый referer (URL, где показывается реклама)
    'geo_id': lambda x: safe_mode(x), # Самая частая геолокация
    'hour': 'mean', # Средний час активности пользователя
    'day_of_week': 'mean', # Средний день недели активности пользователя
    'browser': lambda x: safe_mode(x), # Самый частый браузер
    'os': lambda x: safe_mode(x), # Самая частая операционная система
    'country_id': lambda x: safe_mode(x), # Самая частая страна
    'region_id': lambda x: safe_mode(x), # Самый частый регион
    'timezone_id': lambda x: safe_mode(x), # Самый частый часовой пояс
    'target': 'first' # Целевая переменная, берется первое значение
}).reset_index()
print("Агрегация по пользователям завершена")

# Удаление строк с пропущенными целевыми значениями
user_features = user_features.dropna(subset=['target'])

# Подготовка данных для модели, разделение на признаки и целевую переменную
X = user_features.drop(['user_id', 'target'], axis=1) # Удаление User_id и target, чтобы остались только признаки
y = user_features['target'] # Создание вектора целевой переменной y. Это то, что модель будет учиться предсказывать

# Преобразование категориальных признаков в числовые коды
for col in ['referer', 'geo_id', 'browser',
            'os', 'country_id', 'region_id',
            'timezone_id']:
    X[col] = X[col].astype('category').cat.codes

# Разделение данных на обучающую и валидационную выборки в пропорции 80/20
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Данные разделены на обучающую и проверочную выборки")

# Создание и обучение модели Random Forest
model = RandomForestClassifier(random_state=42, n_estimators=100) # Параметры разбиения данных и количество деревьев
model.fit(X_train, y_train)
print("Модель обучена")

# Сохранение модели в файл random_forest_model.joblib
joblib.dump(model, 'random_forest_model.joblib')
print("Модель сохранена в файл random_forest_model.joblib")


