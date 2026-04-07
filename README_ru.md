**[English version](README.md)**

# Оптимизатор кварцевых полосовых фильтров

Инструмент для **подбора параметров лестничных полосовых кварцевых фильтров** так, чтобы их **АЧХ была близка к характеристике идеального фильтра** даже при наличии потерь в резонаторах. Программа позволяет скомпенсировать влияние неидеальности кварцевых резонаторов, которое обычно не учитывается в методах типа Dishal и приводит к значительному искажению формы АЧХ.

**Репозиторий:** [github.com/MatthewMih/crystal-rf-filter-optimizer](https://github.com/MatthewMih/crystal-rf-filter-optimizer)  
**Автор:** Matvey Mikhalchuk — [mikhalchuk.matvey@gmail.com](mailto:mikhalchuk.matvey@gmail.com)  
**Полная техническая документация (русский):** [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) · [English — DOCUMENTATION_en.md](docs/DOCUMENTATION_en.md)

---

## Зачем это нужно

В классических методах расчёта лестничных фильтров (в духе **Dishal** и аналогов) кварцы часто считаются **идеальными**: бесконечная добротность, без потерь. У реального резонатора конечная **добротность** и **моциональное сопротивление** \(R_m\) (модель **BVD**). Из‑за этого **полоса сужается**, **равномерность в полосе** заметно ухудшается, появляются **дополнительные потери** по сравнению с идеальной кривой.

Эта программа позволяет:

1. Построить **эталонную** АЧХ из JSON-схемы (часто идеальные или номинальные кварцы). Для этого нужно задать номиналы идеальной схемы, параметры резонаторов (в них положить последовательное сопротивление R=0)  
2. Описать **более реалистичную** схему (та же топология, оптимизируемые номиналы конденсаторов, подбор входного/выходного импедансов, зафиксированные параметры кварцев (нужно задать параметры ваших резонаторов, как в п.1, только с учетом R)).  
3. Запустить **градиентную оптимизацию** (Adam, опционально LBFGS), чтобы отклик **неидеальной** схемы **по форме** приблизился к эталону в децибелах на выбранной сетке частот.

Разумные начальные номиналы и физическая реализуемость по‑прежнему на вас; оптимизатор **подстраивает заданные параметры**, а не заменяет теорию фильтров.

---

## Пример (после оптимизации)

Изменение АЧХ в процессе оптимизации номиналов конденсаторов 10.7 МГц фильтра Чебышева.

- **target** (зелёная) — АЧХ идеального фильтра без потерь  
- **target shifted** (оранжевая) — тот же эталон со сдвигом по частоте/уровню (обучаемый)  
- **current** (синяя) — АЧХ реальной схемы с потерями в процессе оптимизации
(подробности в [документации](docs/DOCUMENTATION.md)).

![Процесс оптимизации фильтра с потерями к АЧХ идеального эталона](docs/assets/ladder_optimization_example.gif)

*Анимация: конфиг `examples/ladder_optimize.json` (750 шагов, `shifted_pred_max_decay`, `slope_db: 20`); после быстрого старта ниже тот же вид — `examples/ladder_run/optimization_yzoom.gif`. В `docs/assets/` лежит снимок этого прогона.*

**Было → стало (узкий масштаб по оси Y):** серая пунктирная **initial (pre-opt)** — АЧХ с номиналами и конечным \(R_m\) *до* оптимизации (типичный результат, если собрать фильтр по номиналам в духе Dishal без учёта потерь в кварцах); **final** (синяя) — *после* подгонки к эталону. Зелёная и оранжевая — идеальный эталон без потерь (без сдвига и со сдвигом). Окно по оси Y задаёт скрипт `rebuild_optimization_gif_yzoom.py`.

![Yzoom: исходная и итоговая АЧХ и эталоны](docs/assets/ladder_final_yzoom.png)

*Кадр из `examples/ladder_run/final_yzoom.png`; копия в `docs/assets/` для отображения на GitHub.*

---

## Возможности (кратко)

- Схема в **JSON**: узлы, ветви, общие именованные параметры.  
- Элементы: `Resistor`, `Capacitor`, `Inductor`, `Impedance`, `VoltageSource`, **`Crystal` (BVD: Rm, Lm, Cm, Cp)**, `CrystalLCC`.  
- Уравнения **MNA**, расчёт по частоте пакетом, `torch.linalg.solve`, автоградиенты по параметрам.  
- Режим **`target`:** `target.npz` и график.  
- Режим **`optimize`:** L1/L2 по dB, **веса по частоте**, сдвиги эталона `delta_f` / `delta_y`, GIF и снимки параметров.  
- Скрипт [`scripts/rebuild_optimization_gif_yzoom.py`](scripts/rebuild_optimization_gif_yzoom.py) — пересборка **приближенного** GIF по оси Y.

---

## Зависимости

Python ≥ 3.10, PyTorch 2.x, NumPy, Matplotlib, Pillow (`requirements.txt`).

---

## Установка

```bash
cd crystal-rf-filter-optimizer
python3 -m pip install -e .
```

---

## Быстрый старт (лестница ~10.7 МГц)

Из корня репозитория:

```bash
python3 -m xtal_filters target \
  --config examples/ladder_ideal.json \
  --out examples/ladder_target \
  --device cpu

python3 -m xtal_filters optimize \
  --config examples/ladder_optimize.json \
  --target examples/ladder_target/target.npz

python3 scripts/rebuild_optimization_gif_yzoom.py \
  --config examples/ladder_optimize.json \
  --run-dir examples/ladder_run \
  --save-final examples/ladder_run/final_yzoom.png
```

Используйте **`optimization.device`: `cpu` или `cuda`**.

---

## Документация

| Файл | Содержание |
|------|------------|
| [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) | Формат JSON, секции `response` / `optimization`, режимы весов, dBm, артефакты, API (русский) |
| [docs/DOCUMENTATION_en.md](docs/DOCUMENTATION_en.md) | Тот же справочник на английском |
| [README.md](README.md) | Эта страница по-английски |

---

## Лицензия

[MIT](LICENSE)
