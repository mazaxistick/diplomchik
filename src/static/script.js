document.addEventListener('DOMContentLoaded', async () => {
    // === SPLASH SCREEN ===
    const splashScreen = document.getElementById('splash_screen');
    if (splashScreen) {
        setTimeout(() => {
            splashScreen.classList.add('hidden');
        }, 3500); // 3.5 секунды анимации перед скрытием
    }

    // === ЭЛЕМЕНТЫ ===
    const inputs = {
        year: document.getElementById('year'),
        region_selector: document.getElementById('region_selector'),
        grp: document.getElementById('grp'),
        income: document.getElementById('income'),
        unemployment: document.getElementById('unemployment'),
        housing_price: document.getElementById('housing_price'),
        housing_construction_rate: document.getElementById('housing_construction_rate'),
        investment: document.getElementById('investment'),
    };

    const valueDisplays = {
        year: document.getElementById('year_val'),
        grp: document.getElementById('grp_val'),
        income: document.getElementById('income_val'),
        unemployment: document.getElementById('unemp_val'),
        housing_price: document.getElementById('housing_val'),
        housing_construction_rate: document.getElementById('housing_const_val'),
        investment: document.getElementById('invest_val'),
    };

    const resultBox = document.getElementById('prediction_result');
    const trendBox = document.getElementById('trend');

    let regionsDataSet = [];

    // Последний известный год в датасете (исторические данные до этого года включительно)
    const LAST_HISTORY_YEAR = 2024;

    // === 1. Загрузка метрик модели ===
    try {
        const r = await fetch('/get_metrics');
        const d = await r.json();
        if (d.status === 'success' && d.metrics) {
            const m = d.metrics;
            document.getElementById('metric_r2').textContent = m.r2 !== undefined ? m.r2.toFixed(4) : '—';
            document.getElementById('metric_mae').textContent = m.mae !== undefined ? m.mae.toFixed(2) : '—';
            document.getElementById('metric_rmse').textContent = m.rmse !== undefined ? m.rmse.toFixed(2) : '—';
            document.getElementById('metric_cv').textContent = m.cv_r2_mean !== undefined
                ? `${m.cv_r2_mean.toFixed(4)} ± ${m.cv_r2_std.toFixed(4)}` : '—';
        }
    } catch(e) { console.warn("Не удалось загрузить метрики"); }

    // === 2. Загрузка регионов ===
    try {
        const r = await fetch('/get_regions');
        const d = await r.json();
        if (d.status === 'success') {
            const selectEl = document.getElementById('region_selector');
            selectEl.innerHTML = '<option value="">Выберите регион...</option>';

            regionsDataSet = d.regions;

            d.regions.forEach(reg => {
                const opt = document.createElement('option');
                opt.value = reg.Region_Name;
                opt.innerText = reg.Region_Name;
                selectEl.appendChild(opt);
            });

            new TomSelect("#region_selector", {
                create: false,
                sortField: { field: "text", direction: "asc" }
            });
        }
    } catch(e) { console.error("Ошибка загрузки регионов"); }

    // === ОБНОВЛЕНИЕ ОТОБРАЖЕНИЯ ПОЛЗУНКОВ ===
    const updateDisplays = () => {
        valueDisplays.year.innerText = `${inputs.year.value} г.`;
        valueDisplays.grp.innerText = `${inputs.grp.value} руб.`;
        valueDisplays.income.innerText = `${inputs.income.value} руб.`;
        valueDisplays.unemployment.innerText = `${inputs.unemployment.value}%`;
        valueDisplays.housing_price.innerText = `${inputs.housing_price.value} руб.`;
        valueDisplays.housing_construction_rate.innerText = `${inputs.housing_construction_rate.value} м²`;
        valueDisplays.investment.innerText = `${inputs.investment.value} руб.`;
    };

    // === БЛОКИРОВКА ПОЛЗУНКОВ (для исторических данных) ===
    const lockSliders = (isLocked) => {
        ['grp', 'income', 'unemployment', 'housing_price', 'housing_construction_rate', 'investment'].forEach(key => {
            const el = inputs[key];
            if (isLocked) {
                el.setAttribute('disabled', 'true');
                el.style.opacity = '0.5';
                el.style.cursor = 'not-allowed';
            } else {
                el.removeAttribute('disabled');
                el.style.opacity = '1';
                el.style.cursor = 'pointer';
            }
        });
    };

    // === ОТРИСОВКА РЕЗУЛЬТАТА ===
    const renderData = (val, isHistory, meta) => {
        resultBox.style.opacity = 0;
        trendBox.style.opacity = 0;

        setTimeout(() => {
            resultBox.innerHTML = `${val > 0 ? '+' : ''}${val.toFixed(2)} <span class="unit">на 10 000 чел.</span>`;
            resultBox.className = 'prediction-value';
            trendBox.className = 'trend-indicator';

            resultBox.title = 'Коэффициент миграционного прироста: число приехавших минус уехавших на 10 000 жителей.';

            if (isHistory) {
                resultBox.classList.add('neutral');
                trendBox.innerText = '📖 ФАКТ РОССТАТА';
                trendBox.style.background = 'rgba(56, 189, 248, 0.2)';
                trendBox.style.color = '#38bdf8';
            } else {
                if (val > 6) {
                    resultBox.classList.add('positive');
                    trendBox.innerText = '🚀 ПРОГНОЗ: Значительный приток';
                    trendBox.style.background = 'rgba(74, 222, 128, 0.2)';
                    trendBox.style.color = '#4ade80';
                } else if (val > 1) {
                    resultBox.classList.add('positive');
                    trendBox.innerText = '📊 ПРОГНОЗ: Умеренный приток';
                    trendBox.style.background = 'rgba(74, 222, 128, 0.2)';
                    trendBox.style.color = '#4ade80';
                } else if (val < -6) {
                    resultBox.classList.add('negative');
                    trendBox.innerText = '⚠️ ПРОГНОЗ: Серьёзный отток';
                    trendBox.style.background = 'rgba(248, 113, 113, 0.2)';
                    trendBox.style.color = '#f87171';
                } else if (val < -1) {
                    resultBox.classList.add('negative');
                    trendBox.innerText = '📉 ПРОГНОЗ: Умеренный отток';
                    trendBox.style.background = 'rgba(248, 113, 113, 0.2)';
                    trendBox.style.color = '#f87171';
                } else {
                    resultBox.classList.add('neutral');
                    trendBox.innerText = '⚖️ ПРОГНОЗ: Баланс';
                    trendBox.style.background = 'rgba(255, 255, 255, 0.1)';
                    trendBox.style.color = '#f8fafc';
                }
            }

            resultBox.style.opacity = 1;
            trendBox.style.opacity = 1;
        }, 150);
    };

    // === ОСНОВНАЯ ЛОГИКА ЗАПРОСА ===
    let timeoutId;

    const fetchBrain = async () => {
        const year = parseInt(inputs.year.value);
        const regionName = inputs.region_selector.value;

        if (!regionName || regionName === "") {
            return;
        }

        const foundReg = regionsDataSet.find(r => r.Region_Name === regionName);
        const federalDistrict = foundReg ? foundReg.Federal_District : 'ЦФО';

        try {
            if (year <= LAST_HISTORY_YEAR) {
                // --- ИСТОРИЧЕСКИЕ ДАННЫЕ ---
                lockSliders(true);
                const r = await fetch('/get_history', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ region: regionName, year: year })
                });
                const data = await r.json();
                if (data.status === 'success') {
                    inputs.grp.value = Math.round(data.grp);
                    inputs.income.value = Math.round(data.income);
                    inputs.unemployment.value = data.unemployment;
                    inputs.housing_price.value = Math.round(data.housing_price);
                    inputs.housing_construction_rate.value = data.housing_construction_rate;
                    inputs.investment.value = Math.round(data.investment);
                    updateDisplays();

                    renderData(data.migration_rate, true, {
                        region: regionName, year: year,
                        grp: data.grp, income: data.income,
                        unemployment: data.unemployment,
                        housing_price: data.housing_price,
                        housing_construction_rate: data.housing_construction_rate,
                        investment: data.investment
                    });
                } else {
                    console.warn(`Данные по региону «${regionName}» за ${year} год отсутствуют.`);
                }
            } else {
                // --- ПРОГНОЗ ---
                lockSliders(false);
                const r = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        year: year,
                        federal_district: federalDistrict,
                        region_name: regionName,
                        grp: inputs.grp.value,
                        income: inputs.income.value,
                        unemployment: inputs.unemployment.value,
                        housing_price: inputs.housing_price.value,
                        housing_construction_rate: inputs.housing_construction_rate.value,
                        investment: inputs.investment.value,
                        population: 1000,  // приблизительное значение
                    })
                });
                const data = await r.json();
                if (data.status === 'success') {
                    renderData(data.migration_rate, false, {});
                } else {
                    console.error(`Ошибка прогноза: ${data.message}`);
                }
            }
        } catch(e) {
            console.error('Error:', e);
        }
    };

    const handleChange = () => {
        updateDisplays();
        clearTimeout(timeoutId);
        timeoutId = setTimeout(fetchBrain, 50);
    };

    // === ПРИВЯЗКА СОБЫТИЙ ===
    Object.values(inputs).forEach(input => {
        input.addEventListener('input', handleChange);
    });

    inputs.region_selector.addEventListener('change', handleChange);

    // Инициализация
    updateDisplays();
    setTimeout(handleChange, 300);
});
