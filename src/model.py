import os
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df['Housing_affordability'] = df['Housing_price'] / df['Income_per_capita'].replace(0, 1)
    df['Real_income_index'] = df['Income_per_capita'] / (df['Housing_price'] / 1000).replace(0, 1)

    df = df.sort_values(['Region_Name', 'Year'])
    for col in ['GRP_per_capita', 'Unemployment_rate', 'Migration_rate']:
        df[f'{col}_lag1'] = df.groupby('Region_Name')[col].shift(1)

    df = df.dropna().reset_index(drop=True)

    numeric_cols = [
        'GRP_per_capita', 'Income_per_capita', 'Unemployment_rate',
        'Housing_price', 'Housing_construction_rate', 'Investment_per_capita',
        'Housing_affordability', 'Real_income_index',
        'GRP_per_capita_lag1', 'Unemployment_rate_lag1', 'Migration_rate_lag1'
    ]

    target = 'Migration_rate'
    y = df[target].values

    # Regional Scalers
    scalers = {}
    scaled_num_list = []
    
    for region in df['Region_Name'].unique():
        reg_df = df[df['Region_Name'] == region]
        reg_scaler = StandardScaler()
        reg_scaled = reg_scaler.fit_transform(reg_df[numeric_cols])
        scalers[region] = reg_scaler
        
        # Preserve original order
        for i, idx in enumerate(reg_df.index):
            df.loc[idx, 'temp_order'] = len(scaled_num_list)
            scaled_num_list.append(reg_scaled[i])
            
    # We must sort back to original order! Wait, we already sorted by Region_Name and Year.
    # So iterating by Region_Name.unique() matches exactly if we just append?
    # No, unique() keeps first appearance, but df is sorted by Region_Name, so it perfectly matches block by block.
    
    X = np.array(scaled_num_list)
    
    meta = {
        'numeric_cols': numeric_cols,
        'input_dim': X.shape[1]
    }
    
    return X, y, scalers, meta, df

from xgboost import XGBRegressor

def train_model():
    print("Запуск обучения глобальной модели XGBoost с Монотонными Ограничениями...")
    data_path = os.path.join("..", "data", "processed", "real_data.csv")
    X, y, scalers, meta, df = load_and_preprocess_data(data_path)

    monotone_constraints = (
        1,   # GRP
        1,   # Income
        -1,  # Unemp
        -1,  # Housing_price
        1,   # Housing_construction_rate
        1,   # Investment
        -1,  # Housing_affordability
        1,   # Real_income_index
        0,   # GRP_lag
        0,   # Unemp_lag
        0    # Mig_lag
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        monotone_constraints=monotone_constraints,
        random_state=42
    )

    model.fit(X, y)
    
    final_preds = model.predict(X)

    r2 = r2_score(y, final_preds)
    mae = mean_absolute_error(y, final_preds)
    rmse = np.sqrt(mean_squared_error(y, final_preds))

    print(f'\nMetrics on full dataset:')
    print(f'R2 Score: {r2:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')

    export_dir = "export"
    os.makedirs(export_dir, exist_ok=True)
    
    # Сохраняем как sklearn pipeline или просто модель
    joblib.dump(model, os.path.join(export_dir, 'xgboost_global.pkl'))
    joblib.dump(scalers, os.path.join(export_dir, 'scalers_dict.pkl'))

    metrics = {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'input_dim': meta['input_dim'],
        'numeric_cols': meta['numeric_cols']
    }

    with open(os.path.join(export_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nОбучение завершено. XGBoost сохранён.")

if __name__ == "__main__":
    train_model()
