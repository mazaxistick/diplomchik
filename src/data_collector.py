import os
import pandas as pd
import numpy as np

def load_real_data(base_path, dataset_path):
    print("Начало сборки реального датасета из множества файлов...")
    # 1. Загрузка базового списка регионов и их ФО
    df_base = pd.read_csv(base_path)
    regions_map = dict(zip(df_base['Region_Name'], df_base['Federal_District']))
    valid_regions = list(regions_map.keys())

    # 2. Файлы и нужные индикаторы из них (минимум 5 файлов по запросу)
    file_indicator_map = {
        "data_01_socio_economic_102_v20260313.csv": {
            'Migration_rate': 'Коэффициенты миграционного прироста на 10000 человек населения',
            'Population': 'Среднегодовая численность населения'
        },
        "data_08_socio_economic_102_v20260313.csv": {
            'GRP_per_capita': 'Валовой региональный продукт на душу населения'
        },
        "data_10_socio_economic_102_v20260313.csv": {
            'Investment_per_capita': 'Инвестиции в основной капитал на душу населения'
        },
        "data_employment_102_v20260313.csv": {
            'Unemployment_rate': 'Уровень безработицы населения'
        },
        "data_prices_102_v20260313.csv": {
            'Housing_price': 'Средние цены на рынке жилья: Средние цены на вторичном рынке жилья'
        },
        "data_03_socio_economic_102_v20260313.csv": {
            'Income_per_capita': 'Среднедушевые денежные доходы населения'
        },
        "data_14_socio_economic_102_v20260313.csv": {
            'Housing_construction_rate': 'Ввод в действие жилых домов: На 1000 человек населения'
        }
    }

    def read_csv_safe(file_path):
        try:
            return pd.read_csv(file_path, sep=';', on_bad_lines='skip', encoding='utf-8', low_memory=False)
        except:
            return pd.read_csv(file_path, sep=';', on_bad_lines='skip', encoding='cp1251', low_memory=False)

    def normalize_region(name):
        name = str(name).strip()
        name = name.replace('г. ', '').replace('г.', '')
        name = name.replace(' г', '')
        if name == 'Ханты-Мансийский автономный округ - Югра': return 'Ханты-Мансийский АО'
        if name == 'Ямало-Ненецкий автономный округ': return 'Ямало-Ненецкий АО'
        if name == 'Чукотский автономный округ': return 'Чукотский АО'
        if name == 'Ненецкий автономный округ': return 'Ненецкий АО'
        if name == 'Еврейская автономная область': return 'Еврейская АО'
        if name == 'Республика Саха (Якутия)': return 'Респ. Саха (Якутия)'
        if name == 'Кабардино-Балкарская Республика': return 'Кабардино-Балкарская Респ.'
        if name == 'Карачаево-Черкесская Республика': return 'Карачаево-Черкесская Респ.'
        if name == 'Республика Северная Осетия-Алания': return 'Респ. Северная Осетия'
        if name == 'Удмуртская Республика': return 'Удмуртская Республика'
        return name

    all_pivots = []

    for file_name, indicators in file_indicator_map.items():
        file_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_path):
            print(f"ВНИМАНИЕ: Файл {file_name} не найден!")
            continue
            
        print(f"Чтение {file_name}...")
        df = read_csv_safe(file_path)
        
        # Оставляем только нужные индикаторы из этого файла
        df_filtered = df[df['indicator_name'].isin(indicators.values())].copy()
        df_filtered['region_norm'] = df_filtered['object_name'].apply(normalize_region)
        df_filtered = df_filtered[df_filtered['region_norm'].isin(valid_regions)]
        
        pivot = df_filtered.pivot_table(
            index=['region_norm', 'year'], 
            columns='indicator_name', 
            values='indicator_value', 
            aggfunc='mean'
        ).reset_index()
        
        # Переименование в английские названия
        rename_dict = {v: k for k, v in indicators.items()}
        pivot.rename(columns=rename_dict, inplace=True)
        
        all_pivots.append(pivot)

    # 3. Объединение датасетов по регионам и годам
    if not all_pivots:
        print("Ошибка: Не удалось загрузить данные ни из одного файла.")
        return None

    final_df = all_pivots[0]
    for pivot in all_pivots[1:]:
        final_df = pd.merge(final_df, pivot, on=['region_norm', 'year'], how='outer')

    # Добавляем базовые колонки
    final_df['Federal_District'] = final_df['region_norm'].map(regions_map)
    final_df.rename(columns={'region_norm': 'Region_Name', 'year': 'Year'}, inplace=True)
    final_df = final_df.dropna(subset=['Federal_District']) # удаляем если регион не смапился

    # Фильтруем года
    final_df['Year'] = final_df['Year'].astype(int)
    final_df = final_df[(final_df['Year'] >= 2000) & (final_df['Year'] <= 2025)] # Берем шире, с 2000 года

    # 4. Очистка и заполнение пропусков (Forward Fill внутри региона)
    final_df = final_df.sort_values(by=['Region_Name', 'Year'])
    cols_to_fill = []
    for d in file_indicator_map.values():
        cols_to_fill.extend(list(d.keys()))
        
    for col in cols_to_fill:
        if col not in final_df.columns:
            final_df[col] = np.nan
        # Удаляем невозможные аномальные значения
        final_df.loc[final_df[col] <= -10000, col] = np.nan
        final_df[col] = final_df.groupby('Region_Name')[col].ffill().bfill()
        
    # Если остались глобальные NaN
    for col in cols_to_fill:
        final_df[col] = final_df[col].fillna(final_df[col].mean())

    # Ограничения аномалий
    final_df['Unemployment_rate'] = final_df['Unemployment_rate'].clip(0.3, 55.0)
    final_df['Population'] = final_df['Population'].clip(lower=10)

    # Убедимся, что есть колонка Mortgage_rate (пользователь может использовать или нет, но добавим фиксированную если надо, 
    # однако в app.py мы её можем и убрать. Оставим пока константу, чтобы не ломать если кто-то ожидает)
    final_df['Mortgage_rate'] = 9.0

    cols_order = ['Region_Name', 'Federal_District', 'Year'] + cols_to_fill
    final_df = final_df[cols_order]

    return final_df

def generate_dataset():
    base_data_path = os.path.join("..", "data", "raw", "regions_base_2010.csv")
    dataset_path = os.path.join("..", "..", "Возможный датасет")

    if not os.path.exists(dataset_path):
        print(f"Путь к датасету не найден: {dataset_path}")
        return

    df = load_real_data(base_data_path, dataset_path)

    if df is not None:
        out_dir = os.path.join("..", "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "real_data.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig')

        print(f"\nДатасет собран из {len(df.columns) - 3} признаков!")
        print(f"Датасет сохранён: {out_path}")
        print(f"Размерность: {df.shape[0]} строк x {df.shape[1]} столбцов")
        print(f"Регионов: {df['Region_Name'].nunique()}, Период: {df['Year'].min()}-{df['Year'].max()}")

    return df

if __name__ == "__main__":
    generate_dataset()
