"""Build a real, executable notebook; optionally execute with the project Python."""
import argparse
import json
import os
from pathlib import Path
import sys
import nbformat as nbf
from kion.config import Config


def build(cfg, execute=False):
    notebook = nbf.v4.new_notebook()
    cells = []
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source))
    def code(source):
        cells.append(nbf.v4.new_code_cell(source))
    md('''# KION: рекомендации на ручных признаках

Первая часть исследовательского задания. Полный эксперимент выполнен локально: валидационная сетка, фиксация параметров, два итоговых переобучения и отложенная тестовая оценка.

Этот ноутбук **использует реальные сохранённые модели**, проверяет их прогнозы и показывает все этапы. Повторное дорогое обучение не запускается автоматически. Команда полного воспроизведения приведена в конце; детали — в `REPORT.md`.''')
    code('''from pathlib import Path
import os, sys, json
ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / "config.toml").exists())
sys.path.insert(0, str(ROOT))
from kion.config import Config
cfg = Config.load(ROOT / "config.toml")
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = str(cfg.threads)
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(cfg.threads); pa.set_io_thread_count(cfg.threads)
from IPython.display import display, Image, Markdown
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_rows", 30)
pd.set_option("display.precision", 5)
RESULTS = cfg.path("results_dir")
def read_json(name):
    return json.loads((RESULTS / name).read_text())
print("Python:", sys.version.split()[0], "| CPU threads:", cfg.threads)''')
    md('''## Источник и выборка

Использован полный `data_original.zip` фиксированного коммита KION. Это записи **о последних просмотрах пар**, не полный журнал повторных событий. Из случайных 100 000 пользователей сохранена вся доступная история, без сэмплирования отдельных строк. Даты берутся из самих данных.''')
    code('''from kion.protocol import load_prepared, make_snapshot, day_number
from kion.io import sha256
sample = read_json("sample_manifest.json")
audit = read_json("data_audit.json")
events, users, items = load_prepared(cfg)  # SHA-256 всех исходных таблиц проверяется внутри
assert len(events) == sample["sample_rows"]
assert events.user_id.nunique() == sample["sample_users"]
display(pd.DataFrame([
    {"набор": "Полный архив", "пользователей": audit["users"], "строк": audit["rows"]},
    {"набор": "Выборка", "пользователей": events.user_id.nunique(), "строк": len(events)},
]))
print("Даты:", audit["min_date"], "—", audit["max_date"])
print("SHA-256 архива:", sample["archive_sha256"])
display(events.head())''')
    md('''## Задача, время и фиксация до теста

Top-10 фильмов, ещё не наблюдавшихся в доступной истории; позитив — записанная пара в следующие 7 дней. Label 0 не означает отказ после показа. Оцениваются пользователи с историей и будущими целями; cold-start и удержание всей базы этим числом не описываются.

Каждый срез использует только прошлые строки. После выбора параметров валидационный срез включён в итоговое обучение. Число деревьев фиксировано; тест не передавался как `eval_set`.''')
    code('''from kion.final import load_protocol
protocol, frozen = load_protocol(cfg)
assert frozen["selection_split"] == "validation" and not frozen["test_evaluated"]
assert cfg.test_cutoff not in protocol["training_cutoffs"]
display(pd.DataFrame([
    {"срез": cutoff, "конец labels (не включён)": str(pd.Timestamp(cutoff) + pd.Timedelta(days=cfg.horizon_days))[:10],
     "роль": "train/refit" if cutoff != cfg.test_cutoff else "held-out test"}
    for cutoff in protocol["training_cutoffs"] + [cfg.test_cutoff]
]))
display(pd.Series(frozen["parameters"], name="Параметры зафиксированы до теста"))
print("Зафиксированный контроль:", protocol["reference_method"])
print("Фиксация UTC:", protocol["locked_at_utc"])''')
    md('''## Исторические признаки и кандидаты

Четыре источника: популярность всей истории, популярность за 7 дней, ItemKNN, жанровый профиль. До 100 от источника, первые 10 каждого сохраняются, затем RRF до 200; прошлые пары исключены. Позитивы из будущего в пул не вставляются.

82 признака: статистики пользователя и фильма за 7/30/90 дней и всю историю, давность, тренды, категории и косвенные связи с жанрами, режиссёрами, типом контента, а также баллы источников. Средняя длительность считается после log1p. Подробные определения — в `results/FEATURES.md`.''')
    code('''from kion.features import FeatureStore
from kion.candidates import CandidateRetriever
from kion.dataset import label_pairs
snapshot = make_snapshot(events, cfg.test_cutoff, cfg.horizon_days)
store = FeatureStore(snapshot, users, items, cfg)
assert store.history.day.max() < day_number(cfg.test_cutoff)
demo_user = min(snapshot.targets)
pairs = CandidateRetriever(store, cfg).generate([demo_user])
features = store.transform(pairs)
assert "label" not in features.columns
seen = set(snapshot.history.loc[snapshot.history.user_id == demo_user, "item_id"])
assert not seen.intersection(pairs.item_id)
print("Пользователь:", demo_user, "| кандидатов:", len(pairs), "| максимальная дата истории:", store.history.last_watch_dt.max())
display(features[["user_id", "item_id", "u_count_7", "i_count_7", "pair_genre_share_all", "cand_knn_score"]].head(10))
# Только после расчёта признаков присоединяем будущие labels для анализа.
example_labels = label_pairs(pairs, snapshot.targets)
print("Позитивов в пуле:", int(example_labels.sum()), "| всего будущих целей:", len(snapshot.targets[demo_user]))''')
    md('''## Валидация и выбор модели

Сетка depth={6,8}, l2={3,10}, не более 300 итераций. Конфигурация выбрана по внешней macro NDCG@10 с полным целевым списком. Внутренняя NDCG CatBoost нормирована в кандидатах и даёт 1 группам без позитивов; она используется для выбора числа деревьев и не заменяет внешнюю оценку.

Первая конфигурация использовала проверенный префикс ранней длинной модели. Поздние итерации исключены из выбора; исходная модель не участвовала в тесте. `execution_kind` и `prior_source_seconds` разделяют перенос префикса и обучение с нуля.''')
    code('''grid = pd.read_csv(RESULTS / "grid_results.csv")
validation = pd.read_csv(RESULTS / "validation_results.csv")
display(grid[["name", "depth", "l2_leaf_reg", "iterations", "ndcg@10", "seconds", "execution_kind", "prior_source_seconds"]])
display(validation)
display(Image(filename=str(RESULTS / "validation_ndcg.png")))''')
    md('''## Итоговое переобучение

Полная модель и абляция обучены на четырёх срезах: до 5000 warm-active пользователей на срез, без групп с нулём найденных позитивов **только в обучении**. Вся когорта используется для исторических статистик.

Следующая ячейка вызывает итоговый pipeline. Для завершённого запуска он проверяет сохранённые артефакты и возвращает их без повторного обучения. Реальные замеры обучения показаны в таблице.''')
    code('''from kion.final import run_final, verify_final
run = run_final(cfg)
assert run["test_evaluated"] and run["parameters"] == frozen["parameters"]
receipts = pd.DataFrame([
    {"модель": name, "пар": receipt["training_rows"], "групп": receipt["training_queries"],
     "признаков": len(receipt["identity"]["features"]), "деревьев": receipt["tree_count"],
     "время, с": receipt["seconds"], "test labels в fit": receipt["test_labels_used"],
     "early stopping": receipt["early_stopping"]}
    for name, receipt in run["models"].items()
])
display(receipts)
assert not receipts["test labels в fit"].any() and not receipts["early stopping"].any()''')
    md('''## Проверка загрузки

Загружаем обе модели с диска, заново скорим тестовых кандидатов и проверяем совпадение каждого балла и агрегированных метрик с сохранённой оценкой. Это проверка воспроизводимости, параметры при ней не меняются.''')
    code('''verification = verify_final(cfg)
assert verification["verified"]
display(pd.Series(verification["metrics"], name="Метрики после загрузки"))
print("Допуск:", verification["tolerance"])
print("SHA-256 модели:", verification["model_sha256"])''')
    md('''## Независимый тест и доверительные интервалы

Все методы ранжируют одинаковых кандидатов; пропуски и новые фильмы остаются в знаменателях. Метрики — средние по пользователям. MAP@10 делится на min(10, число целей). Ничьи разрешаются по ID фильма.

Bootstrap: 1000 выборок пользователей с возвращением, seed 42, percentile 95%. Парное сравнение использует одних и тех же пользователей; первичный контроль выбран на валидации. Модели фиксированы: это не интервалы переобучения, других временных периодов или онлайн-эффекта. Вторичные сравнения не скорректированы на множественность.''')
    code('''test = pd.read_csv(RESULTS / "test_results.csv")
intervals = read_json("test_confidence_intervals.json")
display(test[["method", "ndcg@10", "ndcg@10_lower", "ndcg@10_upper", "recall@10", "map@10", "candidate_recall@200"]])
paired = pd.DataFrame([
    {"CatBoost минус": method, **values["ndcg@10"]}
    for method, values in intervals["paired_catboost_minus"].items()
])
display(paired)
print("Первичное сравнение:", intervals["reference_selected_on_validation"])''')
    code('''display(Image(filename=str(RESULTS / "test_metrics.png")))
display(Image(filename=str(RESULTS / "test_paired_differences.png")))''')
    md('''## Абляция и ошибки извлечения кандидатов

В абляции удалены pair_*, cand_knn_*, cand_genre_* и rrf_score. Сохраняются пользовательские/объектные характеристики и персональный пул; это не полное отключение персонализации.

Покажем группы по длине истории и верхнюю границу качества при идеальном порядке имеющихся кандидатов. Oracle не является обученной моделью.''')
    code('''segments = pd.read_csv(RESULTS / "test_segments.csv")
selected = segments[segments["method"].isin(["catboost", run["reference_method"], "catboost_without_personal_features"])]
display(selected[["segment", "method", "users", "ndcg@10", "recall@10"]])
from scripts.build_report import oracle_ndcg
per_user = pd.read_parquet(ROOT / run["artifact_directory"] / "catboost_per_user.parquet")
print("Без единого позитивного кандидата:", int((per_user["candidate_recall@200"] == 0).sum()), "из", len(per_user))
print("Oracle NDCG@10 на этом пуле:", round(oracle_ndcg(per_user), 5))''')
    md('''## Признаки и стоимость

PredictionValuesChange описывает изменение предсказаний, не причинный эффект. Коррелирующие признаки делят важность. Общий эффект удалённой группы оценивается отдельной абляцией, а не суммой этих чисел.''')
    code('''importance = pd.read_csv(RESULTS / "final_feature_importance.csv")
display(importance.head(15))
print("Время финального запуска, мин:", round(run["elapsed_seconds"] / 60, 2))
print("Peak RSS, МиБ:", round(run["peak_rss_mb"], 1))
print("Контрольные суммы и версии:")
display(pd.Series(run["versions"]))''')
    md('''## Выдача рекомендаций

Используем итоговую модель на срезе 16 августа. Баллы задают порядок выдачи и не являются вероятностями просмотра.''')
    code('''from kion.inference import recommend_saved
recommendation = recommend_saved(cfg, demo_user, k=10, stage="test")
assert recommendation["as_of"] == cfg.test_cutoff
assert len(recommendation["item_ids"]) == len(set(recommendation["item_ids"]))
assert not seen.intersection(recommendation["item_ids"])
output = pd.DataFrame({"item_id": recommendation["item_ids"], "score": recommendation["scores"]})
display(output.merge(items[["item_id", "genres", "release_year"]], on="item_id", how="left"))''')
    md('''## Ограничения и воспроизведение

Одна когорта и одна неделя не устанавливают качество на всей базе. Last-watch формат не восстанавливает историю повторных просмотров; неизвестные версии метаданных считаются статическими. Нет показов и истинных негативов; warm-active оценка не измеряет удержание и качество cold-start. Офлайн-прирост не равен причинному бизнес-эффекту.

Для второй части baseline фиксирован. Альтернатива выбирается по train/validation до её тестовой оценки и сравнивается на том же сохранённом тесте. Если тестовые результаты используются для итеративного выбора моделей, для независимого вывода понадобится новый holdout.

Полный повтор обучения, с отдельными результатами:

```bash
python -m scripts.reproduce --name reproduction --full
```

Проверка готовых моделей без обучения:

```bash
python main.py verify-test
python main.py recommend --user-id 17 --stage test
```

Команда ниже только печатает инструкцию. Переключатель нужен, чтобы обычное выполнение ноутбука не запускало повторно длительную CPU-сетку.''')
    code('''RETRAIN_FROM_SCRATCH = False
if RETRAIN_FROM_SCRATCH:
    import subprocess
    subprocess.run([sys.executable, "-m", "scripts.reproduce", "--name", "notebook_reproduction", "--full"], cwd=ROOT, check=True)
else:
    print("Полный эксперимент уже выполнен; дорогое обучение повторно не запускается.")
print("Проверено: реальные данные, кандидаты, фиксированные параметры, обе загруженные модели и тестовые прогнозы.")''')
    notebook.cells = cells
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3 (KION)', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': sys.version.split()[0]}}
    folder = cfg.root / 'notebooks'; folder.mkdir(exist_ok=True)
    destination = folder / 'kion_baseline.ipynb'
    nbf.validate(notebook)
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        runtime = cfg.path('artifact_dir') / 'notebook_runtime'
        kernel_folder = runtime / 'kernels' / 'kion'; kernel_folder.mkdir(parents=True, exist_ok=True)
        spec = {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'KION project Python', 'language': 'python'}
        (kernel_folder / 'kernel.json').write_text(json.dumps(spec))
        env = os.environ.copy()
        for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            env[key] = str(cfg.threads)
        for name, subfolder in [('IPYTHONDIR', 'ipython'), ('JUPYTER_RUNTIME_DIR', 'jupyter'), ('MPLCONFIGDIR', 'matplotlib')]:
            path = runtime / subfolder; path.mkdir(parents=True, exist_ok=True); env[name] = str(path)
        manager = KernelManager(kernel_name='kion', connection_file=str(runtime / 'connection.json'),
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime / 'kernels')]))
        client = NotebookClient(notebook, km=manager, timeout=240, allow_errors=False,
                                resources={'metadata': {'path': str(cfg.root)}})
        try:
            client.execute(cwd=str(cfg.root), env=env)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        if any(cell.execution_count is None or any(out.output_type == 'error' for out in cell.outputs) for cell in code_cells):
            raise RuntimeError('Notebook did not execute all code cells successfully')
        notebook.metadata['execution_info'] = {'python_executable': sys.executable, 'code_cells_executed': len(code_cells),
                                               'test_model_sha256': json.loads((cfg.path('results_dir') / 'test_manifest.json').read_text())['models']['catboost']['model_sha256']}
    nbf.validate(notebook)
    nbf.write(notebook, destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config.toml')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(build(Config.load(args.config), execute=args.execute))
