# xtal_filters_optimizer

Исходный код и документация: [github.com/MatthewMih/xtall_filters_optimizer](https://github.com/MatthewMih/xtall_filters_optimizer).

Исследовательский фреймворк на **PyTorch** для расчёта линейных цепей переменного тока по описанию в **JSON**, сравнения АЧХ в **dBm на нагрузке** и **дифференцируемой оптимизации** параметров (Adam, опционально LBFGS) с сохранением графиков и GIF.

## Возможности

- Задание схемы: узлы `GND`, …, ветви между `node1` и `node2`, общий список именованных параметров (одно имя — один скаляр, можно переиспользовать в нескольких элементах).
- Элементы: `Resistor`, `Capacitor`, `Inductor`, `Impedance` (R + jX), `VoltageSource`, `Crystal` (модель **BVD**: Rm, Lm, Cm, Cp), `CrystalLCC` (как в `uni_linalg.ipynb`: параллельное соединение ветвей L–C1 и C2).
- Решатель: **MNA** (как в ноутбуке), `torch.linalg.solve`, батч по частоте, комплексные типы `complex64` / `complex128`.
- Устройство вычислений: `cpu`, `cuda`, `mps`.
- Режим **target**: расчёт эталонной кривой и сохранение `target.npz`.
- Режим **optimize**: подгонка обучаемых параметров под target, сдвиги `delta_f` / `delta_y` для target, L1/L2 по dBm, штраф за `|delta_y|`, линейная интерполяция target, маска вне диапазона частот.

## Установка

Из корня репозитория:

```bash
python3 -m pip install -e .
```

или только зависимости:

```bash
python3 -m pip install -r requirements.txt
```

Нужны: Python ≥ 3.10, PyTorch 2.x, NumPy, Matplotlib, Pillow.

Запуск модулей предполагает, что текущая рабочая директория — корень проекта (или пакет установлен в окружение):

```bash
cd /path/to/xtal_filters_optimizer
python3 -m xtal_filters --help
```

## Структура пакета `xtal_filters/`

| Модуль | Назначение |
|--------|------------|
| `config.py` | Загрузка JSON, базовая проверка полей |
| `parameters.py`, `parametrization.py` | Реестр параметров, softplus + clamp для положительных R/L/C |
| `elements.py` | Импедансы ветвей ω → Z(ω) |
| `circuit.py` | Топология из JSON |
| `mna.py` | Сборка матрицы MNA и правой части |
| `engine.py` | `ACAnalysis`: схема + sweep → кривая dBm |
| `response.py` | Мощность на резисторе нагрузки, перевод в dBm |
| `interp.py` | Сдвинутый target и маска |
| `loss.py` | Ошибка L1/L2 по dBm |
| `optimize.py` | Цикл оптимизации и артефакты |
| `viz.py` | PNG и GIF; опционально фиксированная ось Y (`y_lim`), хелперы для «зума» по идеальному target |
| `loss_weights.py` | Веса loss по частоте (`two_band`, `target_above_db`, `target_level_decay`, `shifted_pred_max_decay`) |
| `target_io.py` | Генерация target с диска |
| `cli.py` | Командная строка |

Вспомогательный скрипт (не пакет): [`scripts/rebuild_optimization_gif_yzoom.py`](scripts/rebuild_optimization_gif_yzoom.py) — пересборка GIF из `params_frames/` с узкой осью Y.

Примеры схем: каталог [`examples/`](examples/) (в т.ч. `ladder_10p696_10p702MHz.json` — идеальная лестница, `ladder_nonideal_opt.json` — неидеал с оптимизацией).

## Формат JSON схемы

### Обязательные поля верхнего уровня

| Поле | Описание |
|------|----------|
| `nodes` | Список имён узлов; обязательно наличие **`GND`**. |
| `voltage_source` | Имя элемента типа `VoltageSource` (источник задаёт ЭДС в вольтах). |
| `load_element` | Имя элемента типа **`Resistor`** — по нему считается мощность и dBm. |
| `parameters` | Список параметров (см. ниже). |
| `elements` | Список ветвей (см. ниже). |
| `sweep` | Частотный sweep (см. ниже). |

### Параметры (`parameters`)

Каждый объект:

- `name` — уникальное имя.
- `value` — начальное значение в СИ (Ом, Ф, Гн, В для `E` и т.д.).
- `trainable` — `true` / `false` (оптимизируются только обучаемые).
- `kind` — `resistance` \| `capacitance` \| `inductance` \| `generic` (для `generic` знак не ограничивается softplus).
- опционально `min`, `max` — после отображения softplus значение **обрезается** в этот интервал.

Один и тот же `name` в полях `params` у разных элементов ссылается на **один** скаляр (общие кварцы, общие Rs/Rl и т.д.).

### Элементы (`elements`)

Каждый объект:

- `type` — один из типов из таблицы ниже.
- `name` — уникальное имя ветви.
- `node1`, `node2` — узлы; **положительное направление тока** в уравнениях: от `node1` к `node2`.
- `params` — словарь имён полей типа → **строка** с именем параметра из `parameters` **или** число (константа).

Набор ключей в `params` строго фиксирован для типа:

| `type` | `params` |
|--------|----------|
| `Resistor` | `R` |
| `Capacitor` | `C` |
| `Inductor` | `L` |
| `Impedance` | `R`, `X` |
| `VoltageSource` | `E` (В) |
| `Crystal` | `Rm`, `Lm`, `Cm`, `Cp` (BVD: последовательное Rm–L–Cm, параллельно Cp) |
| `CrystalLCC` | `L`, `C1`, `C2` |

### Секция `sweep`

| Поле | Описание |
|------|----------|
| `f_min`, `f_max` | Границы частоты, Гц |
| `num_points` | Число точек (линейная сетка, ≥ 2) |
| `complex_dtype` | опционально: `complex64` (по умолчанию) или `complex128` |

### Секция `response` (опционально)

Задаёт, в каких единицах отдаётся кривая **на нагрузке** (`load_element`) во всех расчётах: `ACAnalysis`, `target`, `optimize`, скрипт yzoom.

| Поле | Описание |
|------|----------|
| `relative_to_input_power` | `false` (по умолчанию) — **dBm** мощности на нагрузке. `true` — **разность в dB**: dBm(нагрузка) − dBm(\(P_\mathrm{avail}\)), где \(P_\mathrm{avail} = E^2/(4R)\) — максимальная средняя мощность, которую может отдать Thevenin-генератор с ЭДС \(E\) и внутренним сопротивлением \(R\) при **идеальном согласовании** нагрузки (\(R_\mathrm{load}=R\)). Эквивалентно \(10\log_{10}(P_\mathrm{load}/P_\mathrm{avail})\). |
| `input_series_resistor` | Имя ветви **`Resistor`**, задающего внутреннее сопротивление \(R\) в формуле \(E^2/(4R)\) (тот же параметр, что и в схеме, например `Rs` с `Rport`). Обязателен, если `relative_to_input_power` = `true`. ЭДС \(E\) берётся из элемента `voltage_source` (параметр `E`). |

```json
"response": {
  "relative_to_input_power": true,
  "input_series_resistor": "Rs"
}
```

Эталонный `target.npz` и схема оптимизации должны использовать **одинаковую** секцию `response`, иначе смысл сравнения теряется.

### Секция `optimization` (только для режима подгонки)

Задаётся в том же JSON, что и схема «фильтра №2». Основные поля:

| Поле | Описание |
|------|----------|
| `device` | `cpu` \| `cuda` \| `mps` |
| `lr` | Скорость обучения Adam |
| `num_steps` | Число шагов Adam |
| `log_every` | Интервал записи в лог (каждый k-й шаг) |
| `gif_every` | Интервал кадров GIF |
| `params_snapshot_every` | Каждые N шагов сохранять `params_frames/step_*.json` (0 = выкл.) |
| `loss_type` | `l1` или `l2` (по разнице dBm на сетке) |
| `lambda_y_shift` | Коэффициент штрафа `λ·|delta_y|` (дБ) |
| `enable_delta_f`, `enable_delta_y` | Обучаемые сдвиги target по частоте (Гц) и по оси dBm |
| `adam_then_lbfgs` | После Adam запустить LBFGS (`true`/`false`) |
| `lbfgs_steps`, `lbfgs_lr` | Параметры LBFGS |
| `seed` | Сид RNG (или `null`) |
| `output_dir` | Каталог результатов (если не передан `--out` в CLI) |

Дополнительно может быть `complex_dtype` (как в `sweep`), если нужно переопределить.

### Веса в loss по частоте (`loss_weighting`)

При равномерном усреднении по всей сетке оптимизатор часто сильнее реагирует на **скаты** АЧХ: там большие градиенты по частоте и типично больше вклад в L1/L2. Чтобы **сместить акцент на полосу пропускания**, задайте веса точек по частоте (физически это просто «насколько важна ошибка на этой частоте»).

Опциональный блок в `optimization`:

```json
"loss_weighting": {
  "mode": "two_band",
  "f_pass_min_hz": 10698500,
  "f_pass_max_hz": 10700000,
  "w_pass": 15,
  "w_stop": 1
}
```

- **`two_band`** — вес `w_pass` внутри \([f_\mathrm{pass,min}, f_\mathrm{pass,max}]\), иначе `w_stop`. Удобно, если границы полосы известны (по проекту или по измерению).
- **`target_above_db`** — вес `w_pass` там, где **интерполированный target** выше порога `db_threshold`, иначе `w_stop`. Не требует ручных Гц: полоса пропускания выделяется по форме эталонной кривой.

```json
"loss_weighting": {
  "mode": "target_above_db",
  "db_threshold": -35,
  "w_pass": 15,
  "w_stop": 1
}
```

- **`target_level_decay`** — вес от **уровня эталонной АЧХ** (интерполированный target на сетке оптимизации). На частоте глобального максимума target вес `w_peak` (по умолчанию 1). На каждые **`slope_db`** дБ ниже этого максимума вес умножается на 0.1 (т.е. линейная прогрессия в **логарифмическом** масштабе по оси dBm). Пример: `slope_db: 20` → на 20 dB ниже пика вес 0.1, на 40 dB ниже — 0.01. Параметры: `slope_db` (обязателен, > 0), `w_peak`, `w_min` (нижняя отсечка).

```json
"loss_weighting": {
  "mode": "target_level_decay",
  "slope_db": 20,
  "w_peak": 1.0,
  "w_min": 1e-8
}
```

- **`shifted_pred_max_decay`** — вес на частоте `f` считается от уровня **`max(target_shifted(f), current(f))` в dBm**, а **пик эталона** — **`max target_shifted`** только по точкам, где loss-маска истинна (пересечение с полосой target после сдвига). Формула та же: `w = w_peak · 10^(-(y_peak - level)/slope_db)`, затем clamp. По умолчанию **`slope_db`: 60** — вес **0.1** на **60 dB** ниже пика **target_shifted** (медленнее, чем при 20 dB на декаду). Веса **пересчитываются каждый forward** (зависят от `pred` и обучаемых `delta_f`, `delta_y`).

```json
"loss_weighting": {
  "mode": "shifted_pred_max_decay",
  "slope_db": 60,
  "w_peak": 1.0,
  "w_min": 1e-8
}
```

Рекомендации: начните с `w_pass / w_stop` от **5…20**; порог для `target_above_db` ставьте **ниже** «плато» полосы, но **выше** дна скатов (например −30…−45 dBm, смотря на ваш target). Для `target_level_decay` уменьшите `slope_db`, если нужно сильнее «прижимать» боковые области (быстрее падение веса от пика). Слишком агрессивные веса могут почти выключить скаты из цели.

Модуль: [`xtal_filters/loss_weights.py`](xtal_filters/loss_weights.py).

### Снимки параметров и прогресс

- `params_snapshot_every` (целое, > 0) — каждые N шагов сохранять JSON со всеми физическими параметрами в `output_dir/params_frames/step_XXXXXX.json` (плюс `step`, `loss`, `delta_f_hz`, `delta_y_db`).
- В терминале ход Adam отображается через **tqdm** (полоса `optimize`).

## Отклик и dBm

Средняя мощность на резисторе нагрузки в фазорной постановке согласована с ноутбуком:

\(P = \frac{1}{2}\,\mathrm{Re}(V \cdot I^*)\) (в коде эквивалентная вещественная форма).

**dBm** (мощность относительно 1 мВт):

\(\mathrm{dBm} = 10 \log_{10}(P_{\mathrm{W}} / 10^{-3})\).

Нагрузка может быть любого сопротивления — оно задаётся параметром резистора `load_element`.

При **`response.relative_to_input_power`: true** по оси — **разность** dBm(нагрузка) и dBm(\(E^2/(4R)\)), т.е. отклик относительно **референса согласованного генератора** (не абсолютная мощность в мВт). Подписи: «dB (load − matched gen.)».

## Командная строка

Рабочая директория — корень репозитория (`cd /path/to/xtal_filters_optimizer`).

### Пример: лестничный фильтр ~10.7 MHz (эталон → оптимизация → yzoom)

**1. АЧХ идеального (эталонного) фильтра** — режим `target` пишет `target.npz` (частоты и dBm на нагрузке), график и метаданные:

```bash
python3 -m xtal_filters target \
  --config examples/ladder_10p696_10p702MHz.json \
  --out examples/ladder_10p7MHz_out \
  --device cpu
```

В `examples/ladder_10p7MHz_out/`: `target.npz`, `target_plot.png`, `target_meta.json`.

**2. Оптимизация неидеальной схемы** под этот эталон (в JSON заданы `optimization`, в т.ч. `loss_weighting`, `params_snapshot_every` для последующего yzoom):

```bash
python3 -m xtal_filters optimize \
  --config examples/ladder_nonideal_opt.json \
  --target examples/ladder_10p7MHz_out/target.npz
```

Каталог результата берётся из `optimization.output_dir` в JSON (сейчас `examples/ladder_opt_shifted_weights`), либо переопределите: `--out <каталог>`.

**3. GIF и PNG с узкой осью Y** (верх ≈ max(идеальный target на сетке оптимизации) + запас, низ по умолчанию −20 dBm). Нужны `params_frames/step_*.json` (в конфиге должно быть `params_snapshot_every` > 0):

```bash
python3 scripts/rebuild_optimization_gif_yzoom.py \
  --config examples/ladder_nonideal_opt.json \
  --run-dir examples/ladder_opt_shifted_weights \
  --save-final examples/ladder_opt_shifted_weights/final_yzoom.png
```

По умолчанию GIF: `<run-dir>/optimization_yzoom.gif`. Параметры: `--y-bottom`, `--y-margin`, `--out-gif`, `--duration-ms`, `--device`.

### Другой минимальный пример (два JSON из корня `examples/`)

**Генерация target:**

```bash
python3 -m xtal_filters target --config examples/ideal_filter.json --out examples/target_run --device cpu
```

**Оптимизация:**

```bash
python3 -m xtal_filters optimize \
  --config examples/nonideal_filter.json \
  --target examples/target_run/target.npz \
  --out examples/out_opt
```

Если не указать `--out`, используется `optimization.output_dir` из JSON.

### Типичный конвейер

1. JSON идеальной схемы → `target` → `target.npz` и график АЧХ.
2. JSON схемы с обучаемыми параметрами и секцией `optimization` → `optimize` с путём к `target.npz`.
3. При необходимости узкого масштаба по оси dBm — `rebuild_optimization_gif_yzoom.py` после прогона с сохранением `params_frames/`.

## Артефакты оптимизации (`output_dir`)

| Файл | Содержимое |
|------|------------|
| `target.npz` | Копия входного target |
| `responses.npz` | `freqs_hz`, `initial`, `final` — кривые dBm до и после |
| `parameters_final.json` | Все параметры в физических единицах |
| `trainable.json` | Только обучаемые параметры |
| `optimization_log.json` | История шагов: loss, `delta_f_hz`, `delta_y_db`, … |
| `params_frames/` | При `params_snapshot_every` > 0 — JSON номиналов по шагам (нужны для yzoom-скрипта) |
| `final.png` | Итоговый график: final / ideal target / shifted target |
| `optimization.gif` | Анимация с авто-масштабом по Y |
| `optimization_yzoom.gif` | Не создаётся оптимизатором; появляется после `rebuild_optimization_gif_yzoom.py` |
| `final_yzoom.png` | То же, если передать `--save-final` в скрипт пересборки |

## Программный API

```python
import torch
from xtal_filters import ACAnalysis, generate_target_artifacts, run_optimization
from xtal_filters.config import load_json

cfg = load_json("examples/ideal_filter.json")
model = ACAnalysis(cfg, device=torch.device("cpu"))
f = torch.linspace(2e6, 8e6, 256, dtype=torch.float64)
dbm = model(f)  # (n_freq,) dBm на нагрузке

generate_target_artifacts("examples/ideal_filter.json", "examples/target_run")

cfg2 = load_json("examples/nonideal_filter.json")
run_optimization(cfg2, "examples/target_run/target.npz", "examples/out_opt")

# Лестничный пример (те же пути, что в CLI выше):
# generate_target_artifacts("examples/ladder_10p696_10p702MHz.json", "examples/ladder_10p7MHz_out")
# run_optimization(
#     load_json("examples/ladder_nonideal_opt.json"),
#     "examples/ladder_10p7MHz_out/target.npz",
#     "examples/ladder_opt_shifted_weights",
# )
```

## Замечания

- **CUDA / MPS**: для комплексных `torch.linalg.solve` возможны ограничения версии PyTorch; при ошибке используйте `device: cpu`.
- Сдвиг target: вне исходного диапазона частот target точки исключаются из среднего loss через маску (см. `interp.py`).
- Репозиторий не использует SPICE в цикле оптимизации — только собственный MNA.

## Связанные файлы

- [`uni_linalg.ipynb`](uni_linalg.ipynb) — исходная постановка MNA и пример кварца `CrystalLCC`.
