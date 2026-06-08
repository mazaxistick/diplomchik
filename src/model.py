"""
Нейросетевая модель прогнозирования миграции населения регионов РФ.

Архитектура: полносвязная сеть (MLP) с 3 скрытыми слоями.
Обучение: мини-батчи, Early Stopping, ReduceLROnPlateau.
Оценка: 5-fold кросс-валидация, метрики MAE / RMSE / R^2.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


# ============================================================
# 1. Загрузка и инженерия признаков
# ============================================================
def load_and_preprocess_data(filepath):
    """
    Загружает CSV, создаёт дополнительные признаки,
    кодирует категории и масштабирует числа.
    """
    print("Загрузка данных...")
    df = pd.read_csv(filepath)

    # --- Инженерия признаков ---

    # Доступность жилья: цена 1 кв.м / среднемесячный доход
    df['Housing_affordability'] = df['Housing_price'] / df['Income_per_capita'].replace(0, 1)

    # Реальный доход с учётом стоимости жилья
    df['Real_income_index'] = df['Income_per_capita'] / (df['Housing_price'] / 1000).replace(0, 1)

    # Лаговые переменные (значения за предыдущий год)
    df = df.sort_values(['Region_Name', 'Year'])
    for col in ['GRP_per_capita', 'Unemployment_rate', 'Migration_rate']:
        df[f'{col}_lag1'] = df.groupby('Region_Name')[col].shift(1)

    # Удаляем строки без лагов (первый год каждого региона)
    df = df.dropna().reset_index(drop=True)

    # --- Разделение признаков ---
    target = 'Migration_rate'
    y = df[target].values

    categorical_cols = ['Federal_District']
    numeric_cols = [
        'Year', 'GRP_per_capita', 'Income_per_capita', 'Unemployment_rate',
        'Housing_price', 'Mortgage_rate', 'Investment_per_capita', 'Population',
        'Housing_affordability', 'Real_income_index',
        'GRP_per_capita_lag1', 'Unemployment_rate_lag1', 'Migration_rate_lag1'
    ]

    # One-Hot для федеральных округов
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cat = encoder.fit_transform(df[categorical_cols])

    # Стандартизация числовых признаков
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df[numeric_cols])

    X = np.hstack((scaled_num, encoded_cat))

    print(f"  Размер выборки: {X.shape[0]} наблюдений, {X.shape[1]} признаков")
    print(f"  Целевая переменная: '{target}'")
    print(f"  Числовые: {len(numeric_cols)}, Категориальные (после OHE): {encoded_cat.shape[1]}")

    # Сохраняем метаданные для инференса
    meta = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'input_dim': X.shape[1]
    }

    return X, y, scaler, encoder, meta, df


# ============================================================
# 2. Архитектура нейронной сети
# ============================================================
class MigrationPredictor(nn.Module):
    """
    Полносвязная нейронная сеть (MLP) для регрессии.
    Входной слой → 128 → 64 → 32 → 1 (выходной нейрон — коэффициент миграции).
    """
    def __init__(self, input_dim):
        super(MigrationPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 3. Обучение с Early Stopping
# ============================================================
def train_model():
    data_path = os.path.join("..", "data", "processed", "real_data.csv")
    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        print("Сначала запустите: python data_collector.py")
        return

    X, y, scaler, encoder, meta, df = load_and_preprocess_data(data_path)

    # Разделение: 80% обучение / 20% тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # DataLoader с мини-батчами
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).view(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)

    # Инициализация модели
    model = MigrationPredictor(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5
    )

    # --- Обучение с Early Stopping ---
    epochs = 500
    patience = 40
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"\nОбучение нейросети ({epochs} эпох макс., Early Stopping patience={patience})...")
    print("-" * 65)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count

        # Валидация на тестовой выборке
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t)
            val_loss = criterion(val_preds, y_test_t).item()

        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Эпоха {epoch+1:>4}/{epochs} | "
                  f"Train MSE: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n  Early Stopping на эпохе {epoch+1}!")
            break

    # Загружаем лучшую модель
    model.load_state_dict(best_model_state)

    # ============================================================
    # 4. Оценка модели
    # ============================================================
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy().flatten()

    mae = mean_absolute_error(y_test, test_preds)
    mse = mean_squared_error(y_test, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_preds)

    print("\n" + "=" * 55)
    print("  РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 55)
    print(f"  MAE  (средняя абс. ошибка):   {mae:.2f}")
    print(f"  RMSE (корень из MSE):          {rmse:.2f}")
    print(f"  R^2   (коэфф. детерминации):    {r2:.4f}")
    print("=" * 55)

    # ============================================================
    # 5. Кросс-валидация (5-fold)
    # ============================================================
    print("\n5-fold кросс-валидация...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_cv_train = torch.FloatTensor(X[train_idx])
        y_cv_train = torch.FloatTensor(y[train_idx]).view(-1, 1)
        X_cv_val = torch.FloatTensor(X[val_idx])
        y_cv_val = y[val_idx]

        cv_model = MigrationPredictor(input_dim=X.shape[1])
        cv_opt = optim.Adam(cv_model.parameters(), lr=0.005, weight_decay=1e-5)
        cv_crit = nn.MSELoss()

        cv_dataset = TensorDataset(X_cv_train, y_cv_train)
        cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=True)

        for ep in range(200):
            cv_model.train()
            for xb, yb in cv_loader:
                p = cv_model(xb)
                l = cv_crit(p, yb)
                cv_opt.zero_grad()
                l.backward()
                cv_opt.step()

        cv_model.eval()
        with torch.no_grad():
            cv_preds = cv_model(X_cv_val).numpy().flatten()
        fold_r2 = r2_score(y_cv_val, cv_preds)
        cv_r2_scores.append(fold_r2)
        print(f"  Fold {fold+1}: R^2 = {fold_r2:.4f}")

    print(f"\n  Среднее R^2 по 5 фолдам: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")

    # ============================================================
    # 6. Экспорт модели и препроцессоров
    # ============================================================
    os.makedirs('export', exist_ok=True)
    torch.save(model.state_dict(), 'export/model.pth')
    joblib.dump(scaler, 'export/scaler.pkl')
    joblib.dump(encoder, 'export/encoder.pkl')

    metrics = {
        'mae': round(mae, 2), 'rmse': round(rmse, 2), 'r2': round(r2, 4),
        'cv_r2_mean': round(np.mean(cv_r2_scores), 4),
        'cv_r2_std': round(np.std(cv_r2_scores), 4),
        'input_dim': meta['input_dim'],
        'numeric_cols': meta['numeric_cols'],
        'categorical_cols': meta['categorical_cols'],
    }
    with open('export/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nМодель, скейлер, энкодер и метрики сохранены в 'export/'")


if __name__ == "__main__":
    train_model()
