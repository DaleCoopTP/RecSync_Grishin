"""Render the completed first-part experiment from immutable result files."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from kion.config import Config
from kion.final import verify_files

LABELS = {'popularity_all': 'Популярность: вся история', 'popularity_7d': 'Популярность: 7 дней',
          'itemknn': 'ItemKNN', 'rrf': 'RRF', 'catboost': 'CatBoost',
          'catboost_without_personal_features': 'CatBoost: абляция связей'}


# Text and tables only; figures keep LABELS so their rendered files stay unchanged.
NAMES = {'popularity_all': 'Популярность за всю историю', 'popularity_7d': 'Популярность за 7 дней',
         'itemknn': 'ItemKNN', 'rrf': 'RRF', 'catboost': 'CatBoost',
         'catboost_without_personal_features': 'CatBoost без признаков связи'}
SEGMENTS = {'history_1_2': '1–2 фильма в истории', 'history_3_9': '3–9 фильмов в истории',
            'history_10_plus': '10 и больше фильмов', 'retrieval_zero_hits': 'в пуле нет ни одной цели',
            'retrieval_some_hits': 'в пуле есть хотя бы одна цель'}


def markdown_table(frame, columns, formats=None):
    formats = formats or {}
    lines = ['| ' + ' | '.join(columns.values()) + ' |', '|' + '|'.join('---' for _ in columns) + '|']
    for row in frame.to_dict('records'):
        lines.append('| ' + ' | '.join(formats.get(k, str)(row[k]) for k in columns) + ' |')
    return '\n'.join(lines)


def oracle_ndcg(per_user):
    truth = per_user.target_count.to_numpy(dtype=int)
    hits = np.rint(per_user['candidate_recall@200'].to_numpy() * truth).astype(int)
    cumulative = np.r_[0., np.cumsum(1 / np.log2(np.arange(2, 12)))]
    return float(np.mean(cumulative[np.minimum(10, hits)] / cumulative[np.minimum(10, truth)]))


def build(cfg):
    results = cfg.path('results_dir')
    run = json.loads((results / 'test_manifest.json').read_text())
    verify_files(cfg, run)
    verification = json.loads((results / 'test_reload_verification.json').read_text())
    if not verification['verified'] or verification['model_sha256'] != run['models']['catboost']['model_sha256']:
        raise ValueError('Run verify-test before building the final report')
    source_path = results / 'final_execution_sources.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    if source and source['code_fingerprint'] != run['code_fingerprint']:
        raise ValueError('Execution source manifest mismatch')
    frozen = json.loads((results / 'frozen_validation.json').read_text())
    protocol = json.loads((results / 'final_protocol.json').read_text())
    sample = json.loads((results / 'sample_manifest.json').read_text())
    audit = json.loads((results / 'data_audit.json').read_text())
    validation = pd.read_csv(results / 'validation_results.csv')
    grid = pd.read_csv(results / 'grid_results.csv').sort_values('ndcg@10', ascending=False)
    grid_gap = float(grid.iloc[0]['ndcg@10'] - grid.iloc[1]['ndcg@10']) if len(grid) > 1 else 0.0
    test = pd.read_csv(results / 'test_results.csv')
    ci = json.loads((results / 'test_confidence_intervals.json').read_text())
    segments = pd.read_csv(results / 'test_segments.csv')
    importance = pd.read_csv(results / 'final_feature_importance.csv')
    model = test.set_index('method').loc['catboost']
    reference = run['reference_method']
    base = test.set_index('method').loc[reference]
    delta = ci['paired_catboost_minus'][reference]['ndcg@10']
    ablation = ci['paired_catboost_minus']['catboost_without_personal_features']['ndcg@10']
    per_user = pd.read_parquet(cfg.root / run['artifact_directory'] / 'catboost_per_user.parquet')
    oracle = oracle_ndcg(per_user)
    n = run['metrics']['catboost']['users']
    zero = int((per_user['candidate_recall@200'] == 0).sum())
    gain = (model['ndcg@10'] / base['ndcg@10'] - 1) * 100
    overview = {'validation_ndcg': float(validation.set_index('method').loc['catboost', 'ndcg@10']),
                'test_ndcg': float(model['ndcg@10']), 'reference_method': reference, 'relative_gain_percent': float(gain),
                'paired_ndcg_difference': delta, 'test_users': n, 'zero_positive_candidates_users': zero,
                'oracle_candidate_ndcg': oracle, 'test_evaluated': True}
    from kion.io import write_json
    write_json(results / 'report_summary.json', overview)
    os.environ.setdefault('MPLCONFIGDIR', str(cfg.path('artifact_dir') / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered = test.sort_values('ndcg@10')
    fig, ax = plt.subplots(figsize=(11, 5))
    values = ordered['ndcg@10'].to_numpy()
    errors = np.vstack([np.maximum(0, values - ordered['ndcg@10_lower']), np.maximum(0, ordered['ndcg@10_upper'] - values)])
    ax.barh([LABELS[x] for x in ordered.method], values, xerr=errors, capsize=3,
            color=['#2463a6' if x == 'catboost' else '#9aabbc' for x in ordered.method])
    for j, value in enumerate(values):
        ax.text(float(ordered.iloc[j]['ndcg@10_upper']) + .002, j, f'{value:.5f}', va='center')
    ax.set_xlim(0, ordered['ndcg@10_upper'].max() * 1.23)
    ax.set_xlabel('Macro NDCG@10; 95% bootstrap по пользователям')
    ax.set_title(f'KION: отложенный тест 16–22 августа 2021\n{n:,} пользователей с историей и целевыми просмотрами')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(results / 'test_metrics.png', dpi=160); plt.close(fig)
    rows = []
    for name, metrics in ci['paired_catboost_minus'].items():
        item = metrics['ndcg@10']
        rows.append({'name': LABELS[name], **item})
    pairs = pd.DataFrame(rows).sort_values('mean')
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.errorbar(pairs['mean'], np.arange(len(pairs)), xerr=np.vstack([
        np.maximum(0, pairs['mean'] - pairs['lower']), np.maximum(0, pairs['upper'] - pairs['mean'])]),
        fmt='o', capsize=4, color='#2463a6')
    ax.set_yticks(np.arange(len(pairs)), pairs['name']); ax.axvline(0, color='#666', linestyle='--')
    ax.set_xlabel('NDCG@10 CatBoost минус контроль; парный 95% bootstrap')
    ax.set_title('Разницы на одних и тех же пользователях')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(results / 'test_paired_differences.png', dpi=160); plt.close(fig)
    def grouped(value):
        return f'{value:,}'.replace(',', ' ')

    table = test.copy(); table['method'] = table.method.map(NAMES)
    table['interval'] = [f"[{lo:.5f}; {hi:.5f}]" for lo, hi in zip(table['ndcg@10_lower'], table['ndcg@10_upper'])]
    val_table = validation.copy(); val_table['method'] = val_table.method.map(NAMES)
    fmt = {key: lambda v: f'{v:.5f}' for key in ['ndcg@10', 'recall@10', 'map@10', 'candidate_recall@200']}
    timing = pd.DataFrame([{'model': NAMES[name], 'rows': grouped(receipt['training_rows']), 'queries': grouped(receipt['training_queries']),
                           'features': len(receipt['identity']['features']), 'trees': receipt['tree_count'],
                           'seconds': receipt['seconds']} for name, receipt in run['models'].items()])
    training = pd.DataFrame([{'cutoff': s['cutoff'], 'queries': grouped(s['queries']), 'rows': grouped(s['rows']),
                             'positive_candidates': grouped(s['positive_candidates'])} for s in run['training_snapshots']])
    diagnostic = segments[segments.method.isin(['catboost', reference, 'catboost_without_personal_features'])].copy()
    diagnostic['method'] = diagnostic.method.map(NAMES)
    diagnostic['segment'] = diagnostic.segment.map(SEGMENTS)
    decision = ('Интервал целиком выше нуля.' if delta['lower'] > 0 else
                'Интервал целиком ниже нуля.' if delta['upper'] < 0 else
                'Интервал включает ноль, устойчивой разницы нет.')
    params = frozen['parameters']
    lines = [
        '# Детали baseline: ручные признаки и CatBoost', '',
        'Приложение к [основному отчёту](../REPORT.md): данные, кандидаты, подбор параметров, абляции, итоговый тест и затраты baseline.', '',
        f"На тесте NDCG@10 = **{model['ndcg@10']:.5f}**, Recall@10 = **{model['recall@10']:.5f}**, MAP@10 = **{model['map@10']:.5f}**. "
        f"Это на {gain:.2f}% выше основного контроля ({NAMES[reference].lower()}): разница {delta['mean']:+.5f}, "
        f"парный 95% CI [{delta['lower']:+.5f}; {delta['upper']:+.5f}]. {decision}", '',
        '## Данные и задача', '',
        f"Источник: [KION](https://github.com/irsafilo/KION_DATASET/tree/{frozen['source_commit']}), коммит `{frozen['source_commit'][:8]}`, "
        f"полный архив `data_original.zip` (SHA-256 в [sample_manifest.json](../results/sample_manifest.json)). "
        f"Размер архива: записей {grouped(audit['rows'])}, пользователей {grouped(audit['users'])}, фильмов {grouped(audit['items'])}; "
        f"даты с {audit['min_date']} по {audit['max_date']}.",
        f"Случайно (seed {cfg.seed}) выбрана когорта пользователей ({grouped(sample['sample_users'])}), их история сохранена "
        f"полностью (записей: {grouped(sample['sample_rows'])}). Демографии нет у части пользователей полного архива "
        f"({grouped(audit['users_without_metadata'])}); для них в признаках стоит значение «нет данных». Метаданные статичны "
        'и считаются известными на дату среза.', '',
        'Цель: рекомендовать top-10 из фильмов, которых пользователь ещё не смотрел. Метка 1 означает запись о просмотре пары '
        'пользователь–фильм в следующие 7 дней, независимо от процента досмотра. Метка 0 означает отсутствие записи, а не отказ '
        'после показа: показов в KION нет. Поэтому выбраны YetiRank и метрики ранжирования, а не LogLoss или ROC-AUC.', '',
        '**Особенность данных.** Строка хранит только последний просмотр пары. Если он попал в будущее, ранние просмотры той же '
        'пары не видны в истории, хотя сами признаки считаются строго по времени.', '',
        'Оцениваются пользователи с историей до среза и хотя бы одним просмотром в следующую неделю, так что это не оценка '
        'удержания всей базы. Новые пользователи в оценку не входят; для них CLI выдаёт популярное, но качество такой выдачи '
        'не измерялось.', '',
        '## Разбиение по времени и защита от утечек', '',
        '| История строго до | Неделя меток | Роль |', '|---|---|---|',
        '| 19 июля | 19–25 июля | Обучение |', '| 26 июля | с 26 июля по 1 августа | Обучение |',
        '| 2 августа | 2–8 августа | Обучение |', '| 9 августа | 9–15 августа | Выбор параметров, затем добавлен в итоговое обучение |',
        '| 16 августа | 16–22 августа | Итоговый тест, оценивается один раз |', '',
        'Кандидаты и признаки строятся до присоединения меток. На каждом срезе статистики, профили и ItemKNN используют только '
        'записи с датой строго раньше среза. Автотесты проверяют, что изменение будущих событий не меняет признаки и что '
        'тестовый срез не попадает в обучение.',
        f"До открытия теста зафиксированы параметры (YetiRank, depth={params['depth']}, l2={params['l2_leaf_reg']}, "
        f"{params['iterations']} деревьев, learning_rate={params['learning_rate']}, seed={params['random_seed']}), основной контроль "
        f"({NAMES[reference].lower()}), bootstrap на {cfg.bootstrap_samples} повторов и абляция "
        '([final_protocol.json](../results/final_protocol.json)). После этого по тесту ничего не подбиралось.',
        'Обе итоговые модели обучены на четырёх срезах без `eval_set`, early stopping и выбора числа деревьев по тесту; тестовый '
        'срез строится только после их сохранения. На каждом срезе берутся до 5000 случайных активных пользователей. Запросы '
        'без позитива среди кандидатов исключаются только из обучения, в оценке они остаются с нулём.', '',
        markdown_table(training, {'cutoff': 'Срез итогового обучения', 'queries': 'Пользователей', 'rows': 'Пар', 'positive_candidates': 'Позитивов среди кандидатов'}), '',
        '## Кандидаты и признаки', '',
        'Четыре источника: популярность за всю историю, популярность за 7 дней, ItemKNN и жанровый профиль пользователя. '
        'Каждый предлагает до 100 фильмов; первые 10 от каждого сохраняются, остальные места до 200 заполняются по RRF (k = 60). '
        'Просмотренные фильмы исключены, будущие позитивы в пул принудительно не добавляются.',
        'ItemKNN строится по бинарным взаимодействиям с последними 100 фильмами пользователя: косинусная близость со сжатием '
        '`co / (co + 10)`, до 100 соседей на фильм. Жанровый источник умножает жанровый профиль пользователя на недавнюю '
        'популярность фильма. Каталог ограничен фильмами, которые встречались до среза.',
        '82 признака: характеристики пользователя и фильма; число просмотров, средняя длительность и % досмотра, давность и тренды '
        'за 7/30/90 дней и всю историю; связи пользователя с жанром, режиссёром и типом кандидата; баллы и ранги источников. '
        'ID пользователя в модель не передаётся, ID фильма передаётся как категория. Определения: [FEATURES.md](../results/FEATURES.md).',
        'Абляция удаляет признаки связи пользователя с кандидатом (`pair_*`), баллы и ранги ItemKNN и жанрового источника '
        '(`cand_knn_*`, `cand_genre_*`) и `rrf_score`; остаётся 45 признаков. Характеристики пользователя и фильма и '
        'персональный пул кандидатов при этом сохраняются, так что это проверка явных связей, а не полное отключение персонализации.', '',
        '## Выбор на валидации', '',
        'Сетка: depth ∈ {6, 8}, l2 ∈ {3, 10}, learning_rate = 0.05, до 300 деревьев. Конфигурация выбирается по macro NDCG@10 '
        'со всеми будущими целями. Число деревьев внутри конфигурации определяет внутренняя NDCG CatBoost: она нормируется только '
        'по позитивам в пуле и для запросов без позитивов равна 1, поэтому с внешней NDCG напрямую не сравнима.', '',
        markdown_table(val_table, {'method': 'Метод', 'ndcg@10': 'NDCG@10', 'recall@10': 'Recall@10', 'map@10': 'MAP@10'}, fmt), '',
        f"Выбраны depth={params['depth']}, l2={params['l2_leaf_reg']}, {params['iterations']} деревьев. Две лучшие конфигурации "
        f"различаются на валидации всего на {grid_gap:.6f} NDCG, так что преимущество глубины 8 не доказано.",
        'В первой конфигурации сетки взяты первые 300 деревьев более раннего запуска на 1500 деревьев (6 потоков); в выборе '
        'участвовали только они. Происхождение: [grid_provenance.json](../results/grid_provenance.json). Выбранная '
        'конфигурация и обе финальные модели обучены заново в 2 потока.', '',
        '## Итоговый тест', '',
        f"История до {cfg.test_cutoff}, горизонт {cfg.horizon_days} дней, {grouped(n)} активных пользователей с историей; "
        f"{grouped(run['test_snapshot']['cold_active_users'])} новых пользователей в оценку не входят. "
        f"Пар кандидатов: {grouped(run['test_snapshot']['rows'])}.",
        'Все методы ранжируют один и тот же пул. Цели вне пула и новые фильмы остаются в знаменателях, ничьи разрешаются по ID '
        'фильма. Метрики усреднены по пользователям; Recall@10 делится на все будущие цели, MAP@10 на min(10, число целей), '
        'идеальная DCG строится по полному списку.', '',
        markdown_table(table, {'method': 'Метод', 'ndcg@10': 'NDCG@10', 'interval': '95% CI NDCG', 'recall@10': 'Recall@10', 'map@10': 'MAP@10', 'candidate_recall@200': 'Recall кандидатов'}, fmt), '',
        '![Тестовые метрики](../results/test_metrics.png)', '',
        '![Парные разницы](../results/test_paired_differences.png)', '',
        f"Полная модель против абляции: ΔNDCG = {ablation['mean']:+.5f}, 95% CI [{ablation['lower']:+.5f}; {ablation['upper']:+.5f}]. "
        'Это совместный эффект удалённой группы признаков при тех же кандидатах и параметрах.',
        f"Основной контроль ({NAMES[reference].lower()}) выбран на валидации; остальные сравнения вторичные и без поправки на "
        f"множественность. Bootstrap: {cfg.bootstrap_samples} выборок пользователей с возвращением, seed = {cfg.seed}. Модели "
        'фиксированы, поэтому интервалы не учитывают разброс от переобучения, состава обучающей выборки и смены недели и '
        'ничего не говорят об онлайн-эффекте.', '',
        '## Что ограничивает качество', '',
        f"Recall кандидатов@200 = {model['candidate_recall@200']:.4f}: у {grouped(zero)} из {grouped(n)} пользователей ({zero / n:.1%}) "
        f"в пул не попала ни одна цель, и любое ранжирование даёт им ноль. При идеальном порядке имеющихся кандидатов NDCG@10 "
        f"была бы {oracle:.5f}. Значит, улучшать можно и генерацию кандидатов, и порядок внутри пула.", '',
        markdown_table(diagnostic, {'segment': 'Сегмент', 'method': 'Метод', 'users': 'Пользователей', 'ndcg@10': 'NDCG@10', 'recall@10': 'Recall@10'}, fmt), '',
        'Ниже важность PredictionValuesChange итоговой модели. Она показывает влияние признака на предсказания, а не причинный '
        'вклад; коррелированные признаки делят важность между собой.', '',
        markdown_table(importance.head(12), {'feature': 'Признак', 'importance': 'Важность'}, {'importance': lambda v: f'{v:.3f}'}), '',
        '## Вычислительные затраты', '',
        markdown_table(timing, {'model': 'Модель', 'rows': 'Пар', 'queries': 'Запросов', 'features': 'Признаков', 'trees': 'Деревьев', 'seconds': 'Обучение, с'}, {'seconds': lambda v: f'{v:.1f}'}), '',
        f"Весь финальный этап занял {run['elapsed_seconds'] / 60:.1f} мин, пик памяти {run['peak_rss_mb']:.0f} МиБ, 2 потока, "
        'Apple M4 с 16 ГБ. Параметр CatBoost `used_ram_limit=6gb` ограничивает только сам CatBoost, а не весь процесс. '
        'Время валидационной сетки сюда не входит.',
        'Кандидаты и признаки строятся батчами, совместная встречаемость для ItemKNN считается блоками, полная матрица '
        'пользователь × фильм не строится. Обучающая таблица и Pool CatBoost держатся в памяти, их размер растёт как число '
        'запросов × лимит кандидатов × число признаков. Совместная встречаемость ItemKNN может дорого расти на плотных каталогах. '
        'CLI рекомендаций пересчитывает профили и ItemKNN при каждом вызове; для онлайн-сервиса нужны кэш профилей и индекса, '
        'регулярное обновление и замер задержки.', '',
        '## Ограничения', '',
        'Одна случайная выборка пользователей, одна тестовая неделя, ограниченное число обучающих запросов и небольшая сетка '
        'гиперпараметров. Формат последнего просмотра, неявные негативы, статичные метаданные и оценка только активных '
        'пользователей ограничивают выводы. Офлайн-прирост не гарантирует прироста CTR, удержания или выручки.',
        'Для сравнения альтернатив baseline зафиксирован: им нужны те же срезы, кандидаты и доступная информация, а их параметры '
        'выбираются на обучении и валидации до тестовой оценки. Если тест использовать для итеративного выбора моделей, для '
        'независимого вывода понадобится новый отложенный период.', '',
        '## Воспроизведение и файлы', '',
        'Python ' + run['python'] + ', точные версии в `requirements.lock`, CPU. Данные и модели в Git не хранятся; их пути и '
        'контрольные суммы перечислены в манифестах.', '',
        'Проверка готового результата без обучения:', '', '```bash',
        'python main.py verify-test', 'python main.py recommend --user-id 17 --stage test --k 10',
        'python -m scripts.build_report', 'python -m scripts.build_notebook --execute', '```', '',
        'Полное переобучение в отдельные каталоги, чтобы не перезаписать опубликованный тест:', '',
        '```bash', 'python -m scripts.reproduce --name reproduction --full', '```', '',
        'Команда повторяет валидационную сетку и оба финальных обучения. Повторная оценка того же теста проверяет '
        'воспроизводимость, но нового независимого эксперимента не даёт.', '',
        '[Выполненный ноутбук](../notebooks/kion_baseline.ipynb) использует сохранённые модели и данные: проверяет артефакты и '
        'прогнозы, показывает признаки, кандидатов, подбор и итоговую оценку. Модели в нём не обучаются.', '',
        f"Финальная модель, её SHA-256 и размеры данных лежат в [test_manifest.json](../results/test_manifest.json), исходники "
        f"запуска в [final_execution_sources.json](../results/final_execution_sources.json) (Git-коммит обучения "
        f"`{source.get('git_commit', 'не указан')[:7]}`). Зафиксированные параметры и интервалы: "
        "[frozen_validation.json](../results/frozen_validation.json), [test_confidence_intervals.json](../results/test_confidence_intervals.json).",
    ]
    from scripts.build_feature_ablation_report import section
    position = lines.index('## Итоговый тест')
    lines[position:position] = section(cfg)
    destination = cfg.root / 'docs' / 'baseline_details.md'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config.toml')
    print(build(Config.load(parser.parse_args().config)))
