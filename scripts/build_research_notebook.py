"""Build a research notebook; --execute verifies saved models, never trains."""
import argparse
import json
import os
from pathlib import Path
import sys
import textwrap

from kion_autofeatures.runtime import configure, execution_lock, monitor_process
configure(2)
from kion.io import sha256
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts" / "research_delivery"


def make_notebook():
    cells = []

    def md(value):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(value).strip()))

    def code(value, **metadata):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(value).strip(), metadata=metadata))

    md("""
    # KION: исследование автоматических признаков

    Ноутбук воспроизводит анализ завершённого эксперимента и прогнозы сохранённых
    моделей. Обучение запрещено программно. Нужны локальные данные/кэши/модели:
    при их отсутствии выполнение останавливается, а не запускает обучение.

    Основной результат: auto NDCG@10 ≈ 0.15700, full ≈ 0.15729; преимущество
    не подтверждено. Специальный отбор уступил одному фиксированному random
    в диагностическом сравнении. Это не доказательство универсального преимущества random.
    Метод и тестовый набор после оценки не меняются.
    """)
    code("""
    import os
    from pathlib import Path
    import sys, json, gc
    ROOT = Path.cwd().resolve()
    if ROOT.name == 'notebooks':
        ROOT = ROOT.parent
    assert (ROOT / 'autofeatures.toml').is_file(), 'Run from project root or notebooks/'
    sys.path.insert(0, str(ROOT))
    from kion_autofeatures.runtime import configure
    configure(2)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown, Image
    from catboost import CatBoostRanker, CatBoost
    from kion.io import sha256
    from kion_autofeatures.config import ResearchConfig
    from kion_autofeatures.final import verify, verify_frozen
    from kion_autofeatures.baseline import Baseline, attach_features
    from kion_autofeatures.grammar import enumerate_features, resolve_specs
    from kion_autofeatures.engine import HistoryFeatures
    from kion_autofeatures.model import predict_model
    from kion.protocol import load_prepared, day_number
    fit_attempts = []
    def forbidden_fit(*args, **kwargs):
        fit_attempts.append('fit')
        raise RuntimeError('Training is disabled in this verification notebook')
    CatBoostRanker.fit = forbidden_fit
    CatBoost.fit = forbidden_fit
    research = ResearchConfig.load(ROOT / 'autofeatures.toml')
    D = research.results
    def read(path):
        return json.loads(Path(path).read_text())
    print({'python': sys.version.split()[0], 'threads': research.threads,
           'run': research.run_name, 'training': 'disabled'})
    """)
    md("""
    ## Зафиксированный эксперимент

    Обучение baseline: 19/26 июля и 2 августа; внешняя валидация 9 августа;
    финальное переобучение также включает 9 августа, тест 16–22 августа 2021.
    Все исторические признаки используют даты строго раньше среза. KION хранит
    последнюю дату пары, а не полный журнал просмотров. Метаданные статические.
    Исходный тест baseline был известен до второй части — ограничение исследования.
    """)
    code("""
    for name in ('inputs', 'selection', 'validation', 'frozen', 'test',
                 'test_intervals', 'test_verification', 'validation_verification'):
        assert (D / f'{name}.json').is_file(), f'Missing completed artifact: {name}'
    frozen = verify_frozen(research)
    baseline = Baseline(research)
    original_test = read(ROOT / 'results/test_manifest.json')
    record = read(D / 'test.json')
    selection = read(D / 'selection.json')
    assert record['test_evaluated'] and record['identity']['frozen_seal'] == frozen['seal']
    model_paths = {'full': ROOT / original_test['models']['catboost']['path']}
    model_paths.update({name: research.artifacts / 'test' / name / 'model.cbm'
                        for name in frozen['final_variants']})
    unchanged_paths = [D / 'frozen.json', *model_paths.values()]
    unchanged_paths += [research.artifacts / 'test' / n / 'fit.json' for n in frozen['final_variants']]
    for path in unchanged_paths:
        assert path.is_file(), f'Missing saved artifact: {path}'
    before = {str(p.relative_to(ROOT)): (sha256(p), p.stat().st_mtime_ns) for p in unchanged_paths}
    print('Frozen seal:', frozen['seal'])
    display(pd.DataFrame([record['snapshot_diagnostics']]))
    print('Final training: 2,940,779 pairs / 14,704 queries; test: 2,284,830 pairs / 11,425 users')
    """)
    md("""
    ## Внутренний выбор и состав признаков

    S: 13 сигналов источников; R: 15 статических колонок; M: 54 ручных агрегата.
    Auto = S+R+A (44), full = S+R+M (82); random = S+R+16 случайных агрегатов (44).
    Грамматика 504 признака: 10 путей, 8 окон, count/sum/mean/max,
    recency/share/trend. Два точных дубликата S исключены только на обучении.
    Screen: 120 деревьев depth=6, обучающая PredictionValuesChange; префиксы
    оцениваются на 1000 пользователях 2 августа. Выбирается минимальный размер
    в пределах 0.0001 от лучшего. Внешняя валидация не выбирает новый набор.
    """)
    code("""
    assert len(enumerate_features()) == 504
    assert len(selection['eligible']) == 502
    assert len(selection['selected']) == len(selection['random']) == 16
    assert not set(selection['selected']) & set(selection['random'])
    prefixes = pd.DataFrame([{'A': p['size'], **p['metrics']} for p in selection['prefixes']])
    display(prefixes)
    from kion_autofeatures.search import choose_prefix
    assert choose_prefix(selection['prefixes'], research.tie_tolerance)['size'] == selection['chosen_size'] == 16
    display(pd.read_csv(D / 'selected_features.csv').astype(object).fillna('—'))
    display(pd.DataFrame([read(D / 'selection_summary.json')['operations']]))
    """)
    md("""
    ## Пересчёт агрегатов на реальной истории

    Для иллюстрации берём первого по сохранённому порядку пользователя с ≥2
    целями и хотя бы одной найденной целью. Это специально выбранный пример,
    он не используется для оценки качества или выбора метода. Генератор получает
    только ID пар, полную историю когорты и статические метаданные; labels не передаются.
    В примере проверяем все 32 агрегата объединения A и random, включая пропуски.
    """)
    code("""
    per_user = pd.read_parquet(research.artifacts / 'test/full/per_user.parquet')
    demo_user = int(per_user.loc[(per_user.target_count >= 2) &
                                (per_user['candidate_recall@200'] > 0)].index[0])
    # per_user is indexed by user_id; assert the key explicitly before filtering caches.
    assert per_user.index.name == 'user_id'
    _, test_spec = baseline.snapshots('test')
    pairs_base = pd.read_parquet(ROOT / test_spec['path'], filters=[('user_id', '==', demo_user)])
    assert 0 < len(pairs_base) <= 200
    pairs = pairs_base[['user_id', 'item_id']].copy()
    auto_names = sorted(set(selection['selected'] + selection['random']))
    events, users, items = load_prepared(baseline.cfg)
    engine = HistoryFeatures(events, users, items, test_spec['cutoff'])
    generated = engine.transform(pairs, resolve_specs(auto_names))
    cached = pd.read_parquet(research.artifacts / 'datasets/test/2021-08-16_evaluation.parquet',
                             filters=[('user_id', '==', demo_user)])
    pd.testing.assert_frame_equal(generated[['user_id', 'item_id']],
                                  cached[['user_id', 'item_id']].reset_index(drop=True))
    np.testing.assert_allclose(generated[auto_names], cached[auto_names], rtol=0, atol=1e-6, equal_nan=True)
    del engine
    gc.collect()
    print('User:', demo_user, '; pairs:', len(pairs), '; recomputed columns:', len(auto_names))
    display(generated[['item_id'] + selection['selected'][:5]].head(10))
    """)
    code("""
    poisoned = events.copy()
    future_mask = poisoned.day >= day_number(test_spec['cutoff'])
    assert future_mask.any()
    poisoned.loc[future_mask, ['total_dur', 'watched_pct']] = 0
    poison_engine = HistoryFeatures(poisoned, users, items, test_spec['cutoff'])
    poison_features = poison_engine.transform(pairs, resolve_specs(auto_names))
    pd.testing.assert_frame_equal(generated, poison_features, check_exact=True)
    future_poison_passed = True
    print('Future duration/percentage changed in memory; all generated features identical.')
    del poisoned, poison_engine, poison_features, events, users
    gc.collect()
    """)
    md("""
    ## Прогноз сохранённых моделей на одном общем пуле

    Баллы задают порядок и не являются вероятностями. В top-10 помечены
    зарегистрированные будущие записи только для иллюстрации. Нельзя судить
    об общем качестве по выбранному пользователю. Все варианты проверяются
    по собственному whitelist признаков и схеме категориальных колонок.
    """)
    code("""
    demo_frame = attach_features(pairs_base, generated)
    recommendations = []
    demo_predictions_match = True
    for name, path in model_paths.items():
        columns = record['models'][name]['identity']['columns']
        categorical = [c for c in baseline.frozen['categorical_features'] if c in columns]
        scores = predict_model(path, demo_frame, columns, categorical, research.threads)
        saved = pd.read_parquet(research.artifacts / 'test' / name / 'predictions.parquet',
                                filters=[('user_id', '==', demo_user)])
        np.testing.assert_array_equal(pairs.item_id.to_numpy(), saved.item_id.to_numpy())
        np.testing.assert_allclose(scores, saved.score, rtol=0, atol=1e-10)
        ranked = demo_frame[['item_id', 'label']].assign(score=scores)
        ranked = ranked.sort_values(['score', 'item_id'], ascending=[False, True]).head(10)
        ranked.insert(0, 'variant', name)
        ranked.insert(1, 'rank', range(1, len(ranked) + 1))
        recommendations.append(ranked)
    example = pd.concat(recommendations, ignore_index=True)
    if 'title' in items.columns:
        example = example.merge(items[['item_id', 'title']], on='item_id', how='left', validate='many_to_one')
    display(example)
    del demo_frame, pairs_base, generated, cached, items, saved, ranked, scores, example, recommendations
    gc.collect()
    """)
    md("""
    ## Полная проверка семи валидационных и пяти тестовых моделей

    Проверяются checksum входов и freeze, точный порядок всех пар, прогнозы после
    загрузки и все три внешние метрики (atol=1e-10). Общие полные будущие цели
    включают фильмы вне пула. Результат не считается независимым новым тестом.
    """)
    code("""
    reload_result = verify(research)
    assert reload_result['verified'] == ['inputs', 'search', 'validation', 'frozen', 'test']
    assert reload_result['reload_verified'] and reload_result['test_evaluated']
    assert not fit_attempts
    print(reload_result)
    gc.collect()
    """)
    md("""
    ## Качество и парные интервалы

    Bootstrap: 1000 выборок пользователей, seed=42, фиксированные модели.
    Первичное сравнение auto−full; остальные диагностические без коррекции
    множественных сравнений. Нет вариации обучения или других недель.
    Сначала показаны все заранее заданные валидационные варианты, затем тестовые.
    """)
    code("""
    validation = pd.read_csv(D / 'validation_results.csv')
    test = pd.read_csv(D / 'test_results.csv')
    metrics = ['variant', 'ndcg@10', 'recall@10', 'map@10', 'candidate_recall@200', 'users']
    display(validation[metrics])
    display(test[metrics])
    intervals = read(D / 'test_intervals.json')
    contrasts = {'auto-full (primary)': intervals['primary_auto_minus_full']['ndcg@10']}
    contrasts.update({'auto-' + n: v['ndcg@10'] for n, v in intervals['diagnostic_auto_minus'].items()})
    display(pd.DataFrame(contrasts).T)
    primary = contrasts['auto-full (primary)']
    assert primary['lower'] <= 0 <= primary['upper']
    assert contrasts['auto-random']['upper'] < 0
    assert contrasts['auto-source_raw']['lower'] <= 0 <= contrasts['auto-source_raw']['upper']
    display(Image(filename=str(D / 'report/test_comparison.png')))
    print('No established superiority or equivalence; selected A loses to one prespecified random control.')
    """)
    md("""
    ## Состав, сегменты и затраты

    Выбраны в основном суммы duration и краткосрочные статистики фильма/демографии.
    Один прямой аналог ручной формулы — genre share за 30 дней. Важность A=48.25%
    не равна дополнительному качеству: коррелированные признаки делят важность.
    Малый внутренний срез, корреляции и временная нестабильность — гипотезы
    проигрыша random, не доказанные причины. Отсутствуют компонентные абляции.

    Сегменты ниже описательные, без отдельных CI. Predict измеряет три прогретых
    пакетных вызова на готовых признаках, без retrieval/генерации/Pool. Историческое
    время fit full включает Pool и не является синхронным сравнением. 44 входа
    вместо 82 не гарантируют экономию памяти: бинарная модель auto чуть больше full.
    """)
    code("""
    display(pd.read_csv(D / 'test_segments.csv'))
    cost_rows = []
    for name, model in record['models'].items():
        cost_rows.append({'variant': name, 'columns': len(model['identity']['columns']),
                          'pool_s': model.get('training_pool_seconds'), 'fit_s': model.get('fit_seconds'),
                          'predict_s': model['predict_seconds_median'], 'model_bytes': model['model_bytes']})
    display(pd.DataFrame(cost_rows))
    resources = []
    for stage, audit_name in [('search', 'validation_verification'), ('validate', 'validation_verification'),
                              ('test', 'test_verification')]:
        resource = read(D / f'{audit_name}.json')['resources'][stage]
        resources.append({'stage': stage, 'seconds': resource['seconds'],
                          'peak_tree_rss_mib': resource['peak_tree_rss_mb'],
                          'over_limit': resource['memory_limit_exceeded']})
    display(pd.DataFrame(resources))
    assert not any(r['over_limit'] for r in resources)
    print('Fixed-model comparisons only; no serving-latency or large-scale claim.')
    """)
    md("""
    ## Вывод и воспроизведение

    Автоматический словарь даёт конкурентную точечную оценку, но выбранный отбор
    не доказал пользу сверх случайного набора или прибавку NDCG к S+R на тесте.
    Равенство с baseline не доказано. Для улучшений нужны несколько прошлых folds,
    несколько random/seed и новый holdout; текущий тест нельзя снова использовать
    для независимого подтверждения после настройки. Одна неделя, last-watch,
    warm-active аудитория, статические метаданные и отсутствие экспозиций ограничивают вывод.

    Полный отчёт: `RESEARCH_REPORT.md`, команды: `docs/automatic_features_usage.md`.
    Проверка без обучения из корня проекта:
    ```bash
    python -m pytest -q
    python -m kion_autofeatures verify
    python -m scripts.build_research_report
    python -m scripts.build_research_notebook --execute
    ```
    Полное повторение исследования после baseline: `python -m kion_autofeatures
    reproduce --name reproduction_01` (новое имя). Оно обучает модели; этот ноутбук — только проверяет.
    """)
    code("""
    after = {str(p.relative_to(ROOT)): (sha256(p), p.stat().st_mtime_ns) for p in unchanged_paths}
    assert before == after, 'Frozen method, saved models or fit receipts changed'
    assert not fit_attempts
    delivery_checks = {'fit_attempts': len(fit_attempts), 'frozen_seal': frozen['seal'],
                       'recomputed_real_feature_columns': len(auto_names),
                       'demo_user_id': demo_user, 'future_poison_passed': future_poison_passed,
                       'saved_models_and_fit_receipts_unchanged': before == after,
                       'demo_saved_predictions_match': demo_predictions_match,
                       'full_reload': reload_result}
    print(json.dumps(delivery_checks, sort_keys=True))
    """, tags=["delivery-checks"])
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3 (project .venv)", "language": "python"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})


def worker():
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from scripts.build_research_report import load_evidence
    research, _ = load_evidence()
    for name in ("test_comparison.png", "summary.json"):
        if not (research.results / "report" / name).is_file():
            raise FileNotFoundError("Build the research report before executing the notebook")
    notebook = make_notebook()
    kernel_folder = RUNTIME / "kernels" / "kion_research"
    kernel_folder.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "KION research verification", "language": "python"}
    (kernel_folder / "kernel.json").write_text(json.dumps(spec))
    env = os.environ.copy()
    for name, subfolder in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter"), ("MPLCONFIGDIR", "matplotlib")):
        path = RUNTIME / subfolder
        path.mkdir(exist_ok=True)
        env[name] = str(path)
    manager = KernelManager(kernel_name="kion_research", connection_file=str(RUNTIME / "connection.json"),
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(RUNTIME / "kernels")]))
    client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute(cwd=str(ROOT), env=env)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if [c.execution_count for c in code_cells] != list(range(1, len(code_cells) + 1)):
        raise RuntimeError("Notebook did not execute every cell in order")
    if any(out.output_type == "error" for c in code_cells for out in c.outputs):
        raise RuntimeError("Notebook contains execution errors")
    checks_cell = next(c for c in code_cells if "delivery-checks" in c.metadata.get("tags", []))
    checks = json.loads("".join(out.text for out in checks_cell.outputs if out.output_type == "stream"))
    if checks["fit_attempts"] or not checks["full_reload"]["test_evaluated"]:
        raise RuntimeError("The notebook must only verify completed results")
    notebook.metadata["execution_info"] = {"code_cells_executed": len(code_cells), **checks}
    nbf.validate(notebook)
    destination = ROOT / "notebooks" / "kion_research.ipynb"
    nbf.write(notebook, destination)
    audit = {"notebook_sha256": sha256(destination), "builder_sha256": sha256(Path(__file__)),
             "code_cells_executed": len(code_cells), "checks": checks}
    (research.results / "report" / "notebook_execution.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.execute:
        from kion_autofeatures.config import ResearchConfig
        research = ResearchConfig.load(ROOT / "autofeatures.toml")
        with execution_lock(ROOT):
            result = monitor_process([sys.executable, "-u", "-m", "scripts.build_research_notebook", "--worker"],
                                     RUNTIME / "execution.log", int(research.memory_limit_gib * 1024**3), ROOT, os.environ.copy())
        report_directory = research.results / "report"
        report_directory.mkdir(exist_ok=True)
        (report_directory / "notebook_resources.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if result["returncode"] or result["memory_limit_exceeded"]:
            raise RuntimeError(f"Notebook verification failed; see {RUNTIME / 'execution.log'}")
    else:
        RUNTIME.mkdir(parents=True, exist_ok=True)
        destination = RUNTIME / "kion_research.template.ipynb"
        nbf.write(make_notebook(), destination)
        print(destination)


if __name__ == "__main__":
    main()
