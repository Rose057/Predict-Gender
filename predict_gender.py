# Импорт всех необходимых библиотек
import pandas as pd
import ast
import joblib
from datetime import datetime

# Загрузка данных из файлов csv
test_df = pd.read_csv('test.csv', sep=';')
test_users_df = pd.read_csv('test_users.csv', sep=';')
referer_vectors_df = pd.read_csv('referer_vectors.csv', sep=';')
geo_info_df = pd.read_csv('geo_info.csv', sep=';')
print("Файлы загружены")

# Объединение всех таблиц в одну
test_df = test_df.merge(referer_vectors_df, on='referer', how='left')
test_df = test_df.merge(geo_info_df, on='geo_id', how='left')
print("Все таблицы объединены")

# Функция для разбора user_agent. Она извлекает browser и os из user_agent
def from_user_agent(x):
    try:
        us_ag_dict = ast.literal_eval(x)
        return us_ag_dict.get('browser', 'unknown'), us_ag_dict.get('os', 'unknown')
    except:
        return 'unknown', 'unknown'

# Разбор user_agent с применением функции from_user_agent и создание новых колонок browser и os
test_df[['browser', 'os']] = test_df['user_agent'].apply(lambda x: pd.Series(from_user_agent(x)))

# Функция для извлечения часа
def hour(x):
    try:
        return datetime.fromtimestamp(x).hour
    except:
        return -1

# Извлечение часа и создание новой колонки hour
test_df['hour'] = test_df['request_ts'].apply(hour)

# Функция для извлечения дня недели
def weekday(x):
    try:
        return datetime.fromtimestamp(x).weekday()
    except:
        return -1

# Извлечение дня недели и создание новой колонки day_of_week
test_df['day_of_week'] = test_df['request_ts'].apply(weekday)

# Функция для извлечения наиболее частого значения
def safe_mode(x):
    try:
        return x.mode()[0]
    except:
        return None

# Заполнение пустых полей в region_id
test_df['region_id'] = test_df['region_id'].fillna('unknown')

# Группировка данных по user_id и вычисление всех признаков
user_features = test_df.groupby('user_id').agg({
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
}).reset_index()
print("Агрегация по пользователю завершена")

# Оставляет только пользователей из test_users_df
user_features = user_features[user_features['user_id'].isin(test_users_df['user_id'])]

# Отбрасывает колонку user_id для подачи в модель
X_test = user_features.drop(['user_id'], axis=1)

# Преобразование категориальных признаков в числовые коды
for col in ['referer', 'geo_id', 'browser',
            'os', 'country_id', 'region_id',
            'timezone_id']:
    X_test[col] = X_test[col].astype('category').cat.codes

# Загрузка сохраненной модели из файла random_forest_model.joblib
model = joblib.load('random_forest_model.joblib')
print("Модель загружена")

# Предсказание пола пользователей
preds = model.predict(X_test)

# Формирование DataFrame с результатами предсказания
preds_df = pd.DataFrame({
    'user_id': user_features['user_id'],
    'target': preds.astype(int),
})

# Сортировка пользователей по тому порядку, который был в test_users.csv
preds_df = preds_df.set_index('user_id').loc[test_users_df['user_id']].reset_index()

# Сохранение всех рузельтатов в файл preds.csv
preds_df.to_csv('preds.csv', sep=';', index=False)
print("Файл preds.csv сохранен")


