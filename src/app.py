"""
Flask-приложение: веб-интерфейс для прогнозирования миграции регионов РФ.
Загружает обученную нейросеть PyTorch и предоставляет API для инференса.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# ============================================================
# Загрузка модели и данных при старте сервера
# ============================================================
if not os.path.exists('export/xgboost_global.pkl'):
    print("ОШИБКА: Сначала запустите model.py для обучения модели!")
    exit(1)

# Глобальные переменные для модели и скейлеров
global_model = None
scalers_dict = None

def load_resources():
    global global_model, scalers_dict
    
    global_model = joblib.load('export/xgboost_global.pkl')
    scalers_dict = joblib.load('export/scalers_dict.pkl')

load_resources()

# Загрузка исторических данных
df_history = pd.read_csv('../data/processed/real_data.csv')

# Подготовка списка регионов с метаданными
regions_meta = df_history[['Region_Name', 'Federal_District']].drop_duplicates().to_dict('records')

# Предвычисление лаговых переменных для истории
df_history = df_history.sort_values(['Region_Name', 'Year'])
for col in ['GRP_per_capita', 'Unemployment_rate', 'Migration_rate']:
    df_history[f'{col}_lag1'] = df_history.groupby('Region_Name')[col].shift(1)
df_history = df_history.dropna().reset_index(drop=True)
df_history['Housing_affordability'] = df_history['Housing_price'] / df_history['Income_per_capita'].replace(0, 1)
df_history['Real_income_index'] = df_history['Income_per_capita'] / (df_history['Housing_price'] / 1000).replace(0, 1)


# ============================================================
# Маршруты
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_regions', methods=['GET'])
def get_regions():
    """Возвращает список регионов с федеральными округами."""
    return jsonify({'status': 'success', 'regions': regions_meta})


@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """Возвращает метрики обученной модели."""
    # Загрузка метрик модели
    metrics = {}
    if os.path.exists('export/metrics.json'):
        with open('export/metrics.json', 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    return jsonify({'status': 'success', 'metrics': metrics})


@app.route('/get_history', methods=['POST'])
def get_history():
    """Возвращает исторические данные по выбранному региону и году."""
    data = request.json
    region = data.get('region')
    year = int(data.get('year'))

    record = df_history[(df_history['Region_Name'] == region) & (df_history['Year'] == year)]
    if not record.empty:
        row = record.iloc[0]
        return jsonify({
            'status': 'success',
            'region': region,
            'year': year,
            'grp': float(row['GRP_per_capita']),
            'income': float(row['Income_per_capita']),
            'unemployment': float(row['Unemployment_rate']),
            'housing_price': float(row['Housing_price']),
            'housing_construction_rate': float(row['Housing_construction_rate']),
            'investment': float(row['Investment_per_capita']),
            'population': float(row['Population']),
            'migration_rate': float(row['Migration_rate']),
        })
    return jsonify({'status': 'error', 'message': 'Данные не найдены'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Эндпоинт для предсказания.
    Ожидает JSON с экономическими показателями.
    """
    data = request.json
    try:
        year = float(data['year'])
        grp = float(data['grp'])
        income = float(data['income'])
        unemp = float(data['unemployment'])
        housing_price = float(data['housing_price'])
        housing_construction_rate = float(data['housing_construction_rate'])
        investment = float(data['investment'])
        population = float(data['population'])
        region_name = data.get('region_name', '')

        # Вычисляем производные признаки
        housing_affordability = housing_price / income if income > 0 else 0
        real_income_index = income / (housing_price / 1000) if housing_price > 0 else 0

        # Лаговые переменные — берём из истории
        region_data = df_history[df_history['Region_Name'] == region_name]
        if not region_data.empty:
            last_row = region_data.iloc[-1]
            grp_lag = float(last_row['GRP_per_capita'])
            unemp_lag = float(last_row['Unemployment_rate'])
            mig_lag = float(last_row['Migration_rate'])
        else:
            grp_lag = grp * 0.95
            unemp_lag = unemp * 1.02
            mig_lag = 0.0

        numeric_features = [
            grp, income, unemp,
            housing_price, housing_construction_rate, investment,
            housing_affordability, real_income_index,
            grp_lag, unemp_lag, mig_lag
        ]

        if region_name in scalers_dict:
            scaler = scalers_dict[region_name]
            num_scaled = scaler.transform([numeric_features])
        else:
            scaler = list(scalers_dict.values())[0]
            num_scaled = scaler.transform([numeric_features])

        base_prediction = float(global_model.predict(num_scaled)[0])

        # Гибридная надстройка: чтобы ползунки никогда не зависали (за пределами истории)
        # и строго соблюдали логику (доходы всегда в плюс, цена всегда в минус)
        linear_adj = (
            num_scaled[0][0] * 0.8 +    # GRP (+)
            num_scaled[0][1] * 1.5 +    # Income (+)
            num_scaled[0][2] * -1.2 +   # Unemp (-)
            num_scaled[0][3] * -1.0 +   # Housing Price (-)
            num_scaled[0][4] * 0.5 +    # Construction (+)
            num_scaled[0][5] * 0.6      # Investment (+)
        )

        # Добавляем временной тренд (естественная демографическая убыль РФ)
        # Если год больше 2024, каждый год отнимает 0.3 пункта по умолчанию
        years_ahead = max(0, year - 2024)
        time_trend = years_ahead * -0.3

        prediction = base_prediction + linear_adj + time_trend

        return jsonify({
            'status': 'success',
            'migration_rate': round(prediction, 2),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_region_trend', methods=['POST'])
def get_region_trend():
    """Возвращает историю миграции по региону для графика."""
    data = request.json
    region = data.get('region')
    region_data = df_history[df_history['Region_Name'] == region].sort_values('Year')

    if region_data.empty:
        return jsonify({'status': 'error'})

    trend = []
    for _, row in region_data.iterrows():
        trend.append({
            'year': int(row['Year']),
            'migration': float(row['Migration_rate']),
            'grp': float(row['GRP_per_capita']),
            'unemployment': float(row['Unemployment_rate']),
        })

    return jsonify({'status': 'success', 'trend': trend})


if __name__ == '__main__':
    print("Сервер запущен: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
