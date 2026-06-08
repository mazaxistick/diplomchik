"""
Flask-приложение: веб-интерфейс для прогнозирования миграции регионов РФ.
Загружает обученную нейросеть PyTorch и предоставляет API для инференса.
"""

from flask import Flask, render_template, request, jsonify
import torch
import joblib
import numpy as np
import pandas as pd
import os
import json
from model import MigrationPredictor

app = Flask(__name__)

# ============================================================
# Загрузка модели и данных при старте сервера
# ============================================================
if not os.path.exists('export/model.pth'):
    print("ОШИБКА: Сначала запустите model.py для обучения нейросети!")
    exit(1)

scaler = joblib.load('export/scaler.pkl')
encoder = joblib.load('export/encoder.pkl')

# Загрузка метрик модели
metrics = {}
if os.path.exists('export/metrics.json'):
    with open('export/metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)

input_dim = metrics.get('input_dim', scaler.mean_.shape[0] + sum(len(c) for c in encoder.categories_))
nn_model = MigrationPredictor(input_dim)
nn_model.load_state_dict(torch.load('export/model.pth', weights_only=True))
nn_model.eval()

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
            'mortgage_rate': float(row['Mortgage_rate']),
            'investment': float(row['Investment_per_capita']),
            'population': float(row['Population']),
            'migration_rate': float(row['Migration_rate']),
        })
    return jsonify({'status': 'error', 'message': 'Данные не найдены'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Прогноз миграции нейросетью.
    Принимает экономические показатели и возвращает предсказание.
    """
    data = request.json
    try:
        federal_district = data['federal_district']
        year = float(data['year'])
        grp = float(data['grp'])
        income = float(data['income'])
        unemp = float(data['unemployment'])
        housing_price = float(data['housing_price'])
        mortgage_rate = float(data['mortgage_rate'])
        investment = float(data['investment'])
        population = float(data['population'])

        # Вычисляем производные признаки
        housing_affordability = housing_price / income if income > 0 else 0
        real_income_index = income / (housing_price / 1000) if housing_price > 0 else 0

        # Лаговые переменные — берём из последнего известного года этого региона
        region_name = data.get('region_name', '')
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

        # Формируем вектор числовых признаков (порядок как при обучении)
        numeric_features = [
            year, grp, income, unemp,
            housing_price, mortgage_rate, investment, population,
            housing_affordability, real_income_index,
            grp_lag, unemp_lag, mig_lag
        ]

        cat_encoded = encoder.transform([[federal_district]])
        num_scaled = scaler.transform([numeric_features])

        X_infer = np.hstack((num_scaled, cat_encoded))
        X_tensor = torch.FloatTensor(X_infer)

        with torch.no_grad():
            prediction = nn_model(X_tensor).item()

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
