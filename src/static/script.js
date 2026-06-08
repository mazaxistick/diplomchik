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
        mortgage_rate: document.getElementById('mortgage_rate'),
        investment: document.getElementById('investment'),
    };

    const valueDisplays = {
        year: document.getElementById('year_val'),
        grp: document.getElementById('grp_val'),
        income: document.getElementById('income_val'),
        unemployment: document.getElementById('unemp_val'),
        housing_price: document.getElementById('housing_val'),
        mortgage_rate: document.getElementById('mortgage_val'),
        investment: document.getElementById('invest_val'),
    };

    const resultBox = document.getElementById('prediction_result');
    const trendBox = document.getElementById('trend');
    const adviceBox = document.getElementById('advice_text');

    let regionsDataSet = [];

    // Последний известный год в датасете (исторические данные до этого года включительно)
    const LAST_HISTORY_YEAR = 2023;

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
        valueDisplays.grp.innerText = `${inputs.grp.value} т.р.`;
        valueDisplays.income.innerText = `${inputs.income.value} р.`;
        valueDisplays.unemployment.innerText = `${inputs.unemployment.value}%`;
        valueDisplays.housing_price.innerText = `${inputs.housing_price.value} р.`;
        valueDisplays.mortgage_rate.innerText = `${inputs.mortgage_rate.value}%`;
        valueDisplays.investment.innerText = `${inputs.investment.value} т.р.`;
    };

    // === БЛОКИРОВКА ПОЛЗУНКОВ (для исторических данных) ===
    const lockSliders = (isLocked) => {
        ['grp', 'income', 'unemployment', 'housing_price', 'mortgage_rate', 'investment'].forEach(key => {
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
        adviceBox.style.opacity = 0;

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
                adviceBox.innerHTML = `<b>🔎 Факт:</b> за <b>${meta.year}</b> год коэффициент миграционного прироста в регионе «<b>${meta.region}</b>» составил <b>${val > 0 ? '+' : ''}${val.toFixed(1)}</b> на 10 000 чел.` +
                    `<br><br>ВРП: ${meta.grp} т.р./чел. | Доходы: ${meta.income} руб./мес. | Безработица: ${meta.unemployment}%` +
                    `<br>Жильё: ${meta.housing_price} руб./м² | Ипотека: ${meta.mortgage_rate}%`;
            } else {
                if (val > 6) {
                    resultBox.classList.add('positive');
                    trendBox.innerText = '🚀 ПРОГНОЗ: Значительный приток';
                    trendBox.style.background = 'rgba(74, 222, 128, 0.2)';
                    trendBox.style.color = '#4ade80';
                    adviceBox.innerHTML = `<b>Прогноз нейросети:</b> модель предсказывает <b>значительный приток</b> населения при данных экономических условиях. Высокий ВРП и доходы создают притягивающий эффект.`;
                } else if (val > 1) {
                    resultBox.classList.add('positive');
                    trendBox.innerText = '📊 ПРОГНОЗ: Умеренный приток';
                    trendBox.style.background = 'rgba(74, 222, 128, 0.2)';
                    trendBox.style.color = '#4ade80';
                    adviceBox.innerHTML = `<b>Прогноз нейросети:</b> модель предсказывает <b>умеренный приток</b> населения. Положительная динамика экономических показателей.`;
                } else if (val < -6) {
                    resultBox.classList.add('negative');
                    trendBox.innerText = '⚠️ ПРОГНОЗ: Серьёзный отток';
                    trendBox.style.background = 'rgba(248, 113, 113, 0.2)';
                    trendBox.style.color = '#f87171';
                    adviceBox.innerHTML = `<b>Прогноз нейросети:</b> модель прогнозирует <b>значительный отток</b> населения. Неблагоприятная экономическая ситуация стимулирует миграцию.`;
                } else if (val < -1) {
                    resultBox.classList.add('negative');
                    trendBox.innerText = '📉 ПРОГНОЗ: Умеренный отток';
                    trendBox.style.background = 'rgba(248, 113, 113, 0.2)';
                    trendBox.style.color = '#f87171';
                    adviceBox.innerHTML = `<b>Прогноз нейросети:</b> модель прогнозирует <b>умеренный отток</b> населения.`;
                } else {
                    resultBox.classList.add('neutral');
                    trendBox.innerText = '⚖️ ПРОГНОЗ: Баланс';
                    trendBox.style.background = 'rgba(255, 255, 255, 0.1)';
                    trendBox.style.color = '#f8fafc';
                    adviceBox.innerHTML = `<b>Прогноз нейросети:</b> миграционный баланс в целом <b>стабилен</b>.`;
                }
            }

            resultBox.style.opacity = 1;
            trendBox.style.opacity = 1;
            adviceBox.style.opacity = 1;
        }, 150);
    };

    // === ОСНОВНАЯ ЛОГИКА ЗАПРОСА ===
    let timeoutId;

    const fetchBrain = async () => {
        const year = parseInt(inputs.year.value);
        const regionName = inputs.region_selector.value;

        if (!regionName || regionName === "") {
            adviceBox.innerHTML = "Пожалуйста, выберите регион для начала работы.";
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
                    inputs.mortgage_rate.value = data.mortgage_rate;
                    inputs.investment.value = Math.round(data.investment);
                    updateDisplays();

                    renderData(data.migration_rate, true, {
                        region: regionName, year: year,
                        grp: data.grp, income: data.income,
                        unemployment: data.unemployment,
                        housing_price: data.housing_price,
                        mortgage_rate: data.mortgage_rate,
                    });
                } else {
                    adviceBox.innerHTML = `Данные по региону «${regionName}» за ${year} год отсутствуют в датасете.`;
                }
            } else {
                // --- ПРОГНОЗ НЕЙРОСЕТЬЮ ---
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
                        mortgage_rate: inputs.mortgage_rate.value,
                        investment: inputs.investment.value,
                        population: 1000,  // приблизительное значение
                    })
                });
                const data = await r.json();
                if (data.status === 'success') {
                    renderData(data.migration_rate, false, {});
                } else {
                    adviceBox.innerHTML = `Ошибка прогноза: ${data.message || 'неизвестная ошибка'}`;
                }
            }
        } catch(e) {
            console.error('Error:', e);
            adviceBox.innerHTML = 'Ошибка связи с сервером.';
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
