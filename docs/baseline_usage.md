# Запуск baseline

Пошаговые команды baseline (ручные признаки и CatBoost). Описание метода
и результатов находится в [приложении к отчёту](baseline_details.md), команды
для автоматических признаков в [отдельной инструкции](automatic_features_usage.md).

## Установка

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python main.py --help
python -m pytest
```

Параметры лежат в `config.toml`; пути в нём задаются относительно файла
конфигурации. Данные, кэши и модели сохраняются локально в `data/` и
`artifacts/` и исключены из Git.

## Валидация

```bash
python main.py download        # или download --archive /path/to/data_original.zip
python main.py prepare
python -u main.py run --smoke  # быстрая техническая проверка: 1000 пользователей, короткое обучение
python -u main.py run
python -m scripts.verify_validation > results/reload_verification.json
python -m scripts.freeze_validation
python main.py evaluate        # повторная оценка сохранённой модели на валидации
python main.py recommend --user-id 17 --k 10
```

- `run` (и его синоним `train`) строит три обучающих среза, обучает четыре
  конфигурации CatBoost и абляцию. Если выборка не подготовлена, она создаётся
  автоматически.
- `--smoke` пишет в отдельные каталоги; его результаты не используются для
  выбора параметров.
- `python -u` выводит прогресс CatBoost без буферизации, в том числе в файл.
  Надпись `test:` в этом выводе относится к валидационному `eval_set`
  9 августа, а не к итоговому тесту.
- Завершённые конфигурации сохраняются сразу. Повторный `run` проверяет
  параметры, данные и контрольные суммы и продолжает сетку; незавершённая
  конфигурация обучается заново, абляция тоже. Чтобы проверить готовый
  результат без обучения, используйте `scripts.verify_validation`.
- `run --reuse-prefix-from artifacts/<old-run>` позволяет взять первые деревья
  ранее обученной модели, если все четыре датасета совпадают по SHA-256.
  В опубликованном эксперименте так получена одна конфигурация сетки
  (первые 300 из 1500 деревьев); происхождение записано в
  `results/grid_provenance.json`.
- Кэши кандидатов, признаков и меток адресуются по конфигурации, содержимому
  данных, версиям библиотек и Python и исходникам пайплайна.

## Итоговый тест

```bash
python main.py lock-test
python -u main.py test
python main.py verify-test > results/test_reload_verification.json
python main.py recommend --user-id 17 --stage test --k 10
```

- `lock-test` фиксирует параметры из `results/frozen_validation.json`, основной
  контроль и протокол bootstrap в `results/final_protocol.json`.
- `test` дообучает обе модели (полную и абляцию) на четырёх срезах, включая
  9 августа, без `eval_set` и early stopping. Тестовый срез строится только
  после сохранения моделей. Повторный вызов проверяет готовые результаты без
  обучения.
- `verify-test` заново загружает обе модели и сверяет все тестовые прогнозы
  и метрики с допуском 1e-10.
- `recommend --stage validation` использует модель на 9 августа,
  `--stage test` берёт итоговую модель на 16 августа; `auto` по умолчанию берёт
  итоговую, если она есть. Баллы задают порядок выдачи и не являются
  вероятностями просмотра.

## Дополнительная абляция и отчёт

```bash
python -m scripts.feature_ablation
python -m scripts.build_report              # docs/baseline_details.md
python -m scripts.build_notebook --execute  # notebooks/kion_baseline.ipynb
```

Абляция сравнивает на валидации модели только на сигналах источников и только
на ручных признаках с полной моделью; полная модель не переобучается. Ноутбук
использует сохранённые модели и не обучает их заново.

## Полное воспроизведение в отдельных каталогах

```bash
python -m scripts.reproduce --name reproduction --full
```

Команда создаёт `reproduction_reproduction.toml` и отдельные каталоги в
`results/reproductions/` и `artifacts/reproductions/`; существующее имя не
перезаписывается. Этот путь целиком проверен на 1000 пользователях с коротким
бюджетом обучения.
