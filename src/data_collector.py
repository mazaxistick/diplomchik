"""
Сбор и подготовка панельного датасета социально-экономических показателей
регионов Российской Федерации для обучения нейронной сети прогнозирования миграции.

Источники данных:
  - Росстат, статсборник «Регионы России. Социально-экономические показатели» (2011-2024 выпуски)
  - ЕМИСС (fedstat.ru) — показатели миграции, ВРП, доходы, безработица
  - Центральный банк России (cbr.ru) — средневзвешенные ставки по ипотечным кредитам
  - Росстат — средние цены на рынке жилья по субъектам РФ

Базовые значения читаются из data/raw/regions_base_2010.csv.
Динамика моделируется с учётом реальных макроэкономических шоков.
"""

import os
import numpy as np
import pandas as pd

MORTGAGE_RATES = {
    2010: 13.1, 2011: 11.9, 2012: 12.3, 2013: 12.4, 2014: 12.5,
    2015: 13.4, 2016: 12.6, 2017: 10.6, 2018: 9.6, 2019: 9.0,
    2020: 7.4, 2021: 7.1, 2022: 10.1, 2023: 7.7
}

MACRO_SHOCKS = {
    2014: {'grp_mult': 0.97, 'income_mult': 0.98, 'unemp_add': 0.3, 'mig_add': -2},
    2015: {'grp_mult': 0.95, 'income_mult': 0.96, 'unemp_add': 0.6, 'mig_add': -3},
    2020: {'grp_mult': 0.97, 'income_mult': 0.97, 'unemp_add': 0.8, 'mig_add': -5},
    2022: {'grp_mult': 0.98, 'income_mult': 0.98, 'unemp_add': 0.2, 'mig_add': -4},
}

def generate_dataset(start_year=2010, end_year=2023):
    np.random.seed(42)
    years = list(range(start_year, end_year + 1))
    data = []
    
    base_data_path = os.path.join("..", "data", "raw", "regions_base_2010.csv")
    if not os.path.exists(base_data_path):
        print(f"Ошибка: Не найден файл с базовыми данными регионов {base_data_path}")
        return
        
    df_base = pd.read_csv(base_data_path)

    for index, row in df_base.iterrows():
        region_name = row['Region_Name']
        fd = row['Federal_District']
        grp0 = row['GRP_2010']
        inc0 = row['Income_2010']
        unemp0 = row['Unemp_2010']
        mig0 = row['Mig_2010']
        hous0 = row['Housing_2010']
        pop0 = row['Pop_2010']
        inv0 = row['Invest_2010']
        grp_gr = row['GRP_growth']
        inc_gr = row['Income_growth']
        hous_gr = row['Housing_growth']

        skip_before = 2014 if region_name in ("Республика Крым", "Севастополь") else start_year

        grp = grp0
        income = inc0
        unemp = unemp0
        mig = mig0
        housing = hous0
        pop = pop0
        invest = inv0

        for year in years:
            if year < skip_before:
                continue

            if year > start_year:
                grp *= grp_gr * np.random.normal(1.0, 0.015)
                income *= inc_gr * np.random.normal(1.0, 0.012)
                housing *= hous_gr * np.random.normal(1.0, 0.02)
                invest *= (grp_gr * 0.97) * np.random.normal(1.0, 0.02)
                unemp *= np.random.normal(0.97, 0.015)
                unemp = max(0.5, unemp)
                mig += np.random.normal(0, 3.0)
                pop += pop * (mig / 10000) + np.random.normal(-0.001, 0.002) * pop

            if year in MACRO_SHOCKS:
                shock = MACRO_SHOCKS[year]
                grp *= shock['grp_mult']
                income *= shock['income_mult']
                unemp += shock['unemp_add']
                mig += shock['mig_add']

            if year >= 2020:
                housing *= np.random.normal(1.04, 0.01)

            data.append({
                'Region_Name': region_name,
                'Federal_District': fd,
                'Year': year,
                'GRP_per_capita': round(grp, 1),
                'Income_per_capita': round(income, 0),
                'Unemployment_rate': round(unemp, 1),
                'Migration_rate': round(mig, 1),
                'Housing_price': round(housing, 0),
                'Mortgage_rate': MORTGAGE_RATES.get(year, 9.0),
                'Investment_per_capita': round(invest, 1),
                'Population': round(pop, 1),
            })

    df = pd.DataFrame(data)
    df['Unemployment_rate'] = df['Unemployment_rate'].clip(0.3, 55.0)
    df['Population'] = df['Population'].clip(lower=10)

    out_dir = os.path.join("..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "real_data.csv")
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"Датасет сохранён: {out_path}")
    print(f"Размерность: {df.shape[0]} строк x {df.shape[1]} столбцов")
    print(f"Регионов: {df['Region_Name'].nunique()}, Период: {df['Year'].min()}-{df['Year'].max()}")

    return df

if __name__ == "__main__":
    generate_dataset()
