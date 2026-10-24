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
    table = test.copy(); table['method'] = table.method.map(LABELS)
    table['interval'] = [f"[{lo:.5f}; {hi:.5f}]" for lo, hi in zip(table['ndcg@10_lower'], table['ndcg@10_upper'])]
    val_table = validation.copy(); val_table['method'] = val_table.method.map(LABELS)
    fmt = {key: lambda v: f'{v:.5f}' for key in ['ndcg@10', 'recall@10', 'map@10', 'candidate_recall@200']}
    timing = pd.DataFrame([{'model': LABELS[name], 'rows': receipt['training_rows'], 'queries': receipt['training_queries'],
                           'features': len(receipt['identity']['features']), 'trees': receipt['tree_count'],
                           'seconds': receipt['seconds']} for name, receipt in run['models'].items()])
    training = pd.DataFrame([{'cutoff': s['cutoff'], 'queries': s['queries'], 'rows': s['rows'],
                             'positive_candidates': s['positive_candidates']} for s in run['training_snapshots']])
    diagnostic = segments[segments.method.isin(['catboost', reference, 'catboost_without_personal_features'])].copy()
    diagnostic['method'] = diagnostic.method.map(LABELS)
    decision = ('Интервал первичного парного сравнения целиком выше нуля.' if delta['lower'] > 0 else
                'Интервал первичного парного сравнения целиком ниже нуля.' if delta['upper'] < 0 else
                'Интервал первичного парного сравнения включает ноль; устойчивый разрыв этим сравнением не подтверждён.')
    params = frozen['parameters']
    lines = [
        '# KION: baseline на ручных признаках — итоговый отчёт', '',
        'Выполнена первая часть исследовательского задания: воспроизводимый временной эксперимент с ручными признаками, CatBoost и независимой тестовой оценкой. Исследовательская альтернатива бустингу в эту работу не входит.', '',
        f"На тесте NDCG@10 = **{model['ndcg@10']:.5f}**, Recall@10 = **{model['recall@10']:.5f}**, MAP@10 = **{model['map@10']:.5f}**.",
        f"Относительная разница NDCG к заранее выбранному контролю ({LABELS[reference]}) — **{gain:+.2f}%**; абсолютная разница {delta['mean']:+.5f}, парный 95% интервал [{delta['lower']:+.5f}; {delta['upper']:+.5f}]. {decision}", '',
        '## Данные и задача', '',
        f"Источник: [KION, фиксированный коммит {frozen['source_commit'][:8]}](https://github.com/irsafilo/KION_DATASET/tree/{frozen['source_commit']}). Используется полный `data_original.zip`, SHA-256 `{frozen['archive_sha256']}`.",
        f"В архиве {audit['rows']:,} записей, {audit['users']:,} пользователей, {audit['items']:,} фильмов; фактические даты {audit['min_date']} — {audit['max_date']}. Дата в строках данных имеет приоритет над описанием источника.",
        f"Seed {cfg.seed}: случайно выбраны {sample['sample_users']:,} пользователей без сэмплирования строк; сохранены все {sample['sample_rows']:,} их доступных взаимодействий. ID выборки и три исходных таблицы проверяются по SHA-256.",
        f"У {audit['users_without_metadata']:,} пользователей полного архива нет строки демографии; они сохраняются с признаком отсутствия. Статические метаданные считаются доступными на дату среза; исторических версий нет.", '',
        'Цель — рекомендовать top-10 ещё не наблюдавшихся в доступной истории фильмов. Label=1 означает зарегистрированную пару пользователь–фильм в следующие 7 дней, независимо от процента просмотра. Label=0 означает отсутствие записи, а не отказ после показа: KION не содержит показов. Поэтому выбран YetiRank и метрики ранжирования; LogLoss/ROC-AUC как качество CTR здесь не интерпретируются.', '',
        '**Особенность источника:** строка хранит последний просмотр пары, а не полный поток повторных событий. Протокол моделирует будущие записи о последних просмотрах; он не восстанавливает реальное состояние истории на каждую дату. Это ограничение выводов даже при строгом отсечении строк по времени.', '',
        'Оцениваются пользователи с историей до среза и хотя бы одним целевым просмотром в следующую неделю. Это качество, условное на будущей активности; оно не измеряет удержание всей базы. Новые пользователи исключены из этой оценки и отдельно посчитаны; для них в CLI предусмотрена популярность без заявленного качества на cold-start.', '',
        '## Временной протокол и отсутствие утечек', '',
        '| Признаки по истории строго до | Период labels | Использование |', '|---|---|---|',
        '| 19 июля | 19–25 июля | Обучение |', '| 26 июля | 26 июля – 1 августа | Обучение |',
        '| 2 августа | 2–8 августа | Обучение |', '| 9 августа | 9–15 августа | Выбор параметров, затем добавлен в итоговое обучение |',
        '| 16 августа | 16–22 августа | Единственная отложенная тестовая оценка |', '',
        'Кандидаты и признаки строятся до присоединения будущих labels. На каждом срезе статистики, профили и ItemKNN используют только строки с датой строго раньше среза. Метаданные будущих записей о взаимодействиях в признаки не попадают. Тестами проверены инвариантность к изменениям будущих событий и запрет тестового среза в обучении.',
        f"До теста зафиксированы `{params}`, контроль `{reference}`, bootstrap {cfg.bootstrap_samples} повторов и абляция. Протокол: `results/final_protocol.json`, фиксация в {protocol['locked_at_utc']}. После этого параметры и признаки по тесту не подбирались.",
        'Обе итоговые модели обучены на четырёх исторических срезах без `eval_set`, early stopping и выбора числа деревьев по тесту. Сам тестовый snapshot создаётся после сохранения обеих моделей. На каждом обучающем срезе берутся до 5000 случайных warm-active пользователей; группы без найденного позитивного кандидата удаляются только из обучения. В оценке они остаются с нулём.', '',
        markdown_table(training, {'cutoff': 'Срез итогового обучения', 'queries': 'Пользователей', 'rows': 'Пар', 'positive_candidates': 'Позитивов в кандидатах'}), '',
        '## Кандидаты и ручные признаки', '',
        'Четыре источника: популярность за всю историю, популярность за 7 дней, ItemKNN и жанровый профиль пользователя. Каждый предлагает до 100 фильмов; первые 10 каждого источника сохраняются, остальные места заполняет RRF с константой 60 до лимита 200. Просмотренные пары исключены; позитивы из будущего принудительно не добавляются.',
        'ItemKNN использует бинарные взаимодействия последних 100 фильмов каждого пользователя, cosine co-occurrence со shrinkage `co_count / (co_count + 10)` и до 100 соседей на фильм. Жанровый источник сочетает исторический профиль жанров и недавнюю популярность фильма. Каталог ограничен фильмами, наблюдавшимися до среза.',
        '82 признака: характеристики пользователя и фильма, агрегаты за 7/30/90 дней и всю историю, давность и тренды, жанровые/режиссёрские/типовые связи, баллы и ранги источников. ID пользователя исключён; ID фильма — отдельная категория. Точные определения: [FEATURES.md](results/FEATURES.md). Среднее длительности считается после log1p.',
        'Диагностическая абляция удаляет `pair_*`, `cand_knn_*`, `cand_genre_*`, `rrf_score`: остаются 45 признаков. Характеристики пользователя и фильма и персональный пул кандидатов сохраняются; это проверка группы явных связей и персональных баллов, а не полного отключения персонализации.', '',
        '## Выбор на валидации', '',
        'Сетка: depth ∈ {6, 8}, l2 ∈ {3, 10}, learning_rate=0.05, не более 300 итераций. Конфигурация выбирается по внешней macro NDCG@10 с полным целевым списком. Число деревьев внутри конфигурации определяется внутренней NDCG CatBoost: она нормирована внутри кандидатов и даёт 1 группам без позитивов, тогда как внешняя метрика даёт им 0. Эти численные значения нельзя сравнивать напрямую.', '',
        markdown_table(val_table, {'method': 'Метод', 'ndcg@10': 'NDCG@10', 'recall@10': 'Recall@10', 'map@10': 'MAP@10'}, fmt), '',
        f"Выбраны depth={params['depth']}, l2={params['l2_leaf_reg']}, {params['iterations']} деревьев. Разница двух лучших конфигураций на валидации всего {grid_gap:.6f} NDCG: это критерий выбора, а не доказательство преимущества глубины 8.",
        'В первой конфигурации сетки использован проверенный префикс раннего запуска на 1500 итераций/6 потоках: только первые 300 точек участвовали в выборе. Его происхождение и дополнительное время сохранены в `grid_provenance.json`. Выбранная конфигурация и оба финальных переобучения обучены на 2 потоках. Длинная исходная модель не использована в тестовой оценке.', '',
        '## Итоговый тест', '',
        f"История до {cfg.test_cutoff}; горизонт {cfg.horizon_days} дней. Warm-active пользователей: {n:,}; новых пользователей вне этой оценки: {run['test_snapshot']['cold_active_users']:,}. Пар кандидатов: {run['test_snapshot']['rows']:,}.",
        'Все методы ранжируют один и тот же пул. Пропущенные кандидаты и новые фильмы остаются в знаменателях. Ничьи разрешаются по ID фильма. Метрики усредняются по пользователям; Recall использует все будущие цели, MAP@10 — знаменатель min(10, число целей), идеальная DCG также строится по полному списку.', '',
        markdown_table(table, {'method': 'Метод', 'ndcg@10': 'NDCG@10', 'interval': '95% CI NDCG', 'recall@10': 'Recall@10', 'map@10': 'MAP@10', 'candidate_recall@200': 'Recall кандидатов'}, fmt), '',
        '![Тестовые метрики](results/test_metrics.png)', '',
        '![Парные разницы](results/test_paired_differences.png)', '',
        f"Парное сравнение полной модели с абляцией: ΔNDCG={ablation['mean']:+.5f}, 95% CI [{ablation['lower']:+.5f}; {ablation['upper']:+.5f}]. Это совместный эффект удалённой группы при фиксированных кандидатах и параметрах, а не вклад каждого отдельного признака.",
        f"Первичный контроль `{reference}` выбран на валидации. Остальные парные сравнения диагностические; интервалы не скорректированы на множественные сравнения. Использован percentile bootstrap {cfg.bootstrap_samples} выборок пользователей с возвращением, seed={cfg.seed}. Модели фиксированы: интервалы не учитывают переобучение, изменение состава обучающей выборки или другую календарную неделю и не являются оценкой онлайн-эффекта.", '',
        '## Что ограничивает качество', '',
        f"Macro Recall кандидатов = {model['candidate_recall@200']:.4f}; у {zero:,} из {n:,} пользователей ({zero / n:.1%}) в пул не попал ни один позитив. Любое ранжирование на таком пуле даст им ноль. При идеальном порядке имеющихся кандидатов верхняя граница NDCG@10 составила бы {oracle:.5f}; это диагностический oracle, а не обученная модель.",
        'Значит, можно улучшать и извлечение кандидатов, и порядок внутри пула. По этому эксперименту нельзя приписывать все промахи одному компоненту.', '',
        markdown_table(diagnostic, {'segment': 'Группа', 'method': 'Метод', 'users': 'Пользователей', 'ndcg@10': 'NDCG@10', 'recall@10': 'Recall@10'}, fmt), '',
        'Ниже — важность PredictionValuesChange итоговой модели. Она описывает изменение предсказаний, а не причинную ценность; коррелирующие признаки делят важность. Отдельные эффекты требуют заранее заданных дополнительных абляций на новой валидации.', '',
        markdown_table(importance.head(12), {'feature': 'Признак', 'importance': 'Важность'}, {'importance': lambda v: f'{v:.3f}'}), '',
        '## Вычислительные затраты', '',
        markdown_table(timing, {'model': 'Модель', 'rows': 'Пар', 'queries': 'Групп', 'features': 'Признаков', 'trees': 'Деревьев', 'seconds': 'Обучение, с'}, {'seconds': lambda v: f'{v:.1f}'}), '',
        f"Полный финальный запуск: {run['elapsed_seconds']:.1f} с ({run['elapsed_seconds'] / 60:.1f} мин), peak RSS {run['peak_rss_mb']:.1f} МиБ. Лимит — 2 потока; `used_ram_limit=6gb` у CatBoost не является общесистемным жёстким лимитом памяти. Время валидационной сетки и ранних прерванных попыток сюда не включено. Аппаратная среда выполненного запуска: Apple M4, 16 ГБ RAM.",
        'Кандидаты и признаки строятся батчами, co-occurrence ItemKNN — блоками. Полная таблица пользователь×каталог не материализуется. Обучающая таблица и CatBoost Pool остаются в памяти: стоимость зависит от числа обучающих запросов × лимита кандидатов × числа признаков. Ограничения истории и числа соседей сокращают ItemKNN, но его попарные co-occurrence могут дорого расти на плотных каталогах. Повторный `test` проверяет сохранённые результаты и не переобучает модели. CLI выдачи заново восстанавливает FeatureStore и ItemKNN при вызове: это offline-демонстрация. Для online-сервиса нужны постоянный кэш профилей/индекса, батчевое обновление и отдельные измерения латентности.', '',
        '## Ограничения и следующие шаги', '',
        'Это одна случайная когорта и одна тестовая неделя, ограниченное число обучающих запросов и небольшой поиск гиперпараметров. Точность вне этой когорты/периода не установлена. Last-watch формат, неявные отрицательные метки, неизвестная историческая версия метаданных и warm-active отбор ограничивают интерпретацию. Нельзя трактовать офлайн-прирост как прирост CTR, удержания или выручки.',
        'Для второй части задания baseline теперь фиксирован: альтернативе нужны те же срезы, кандидаты, доступная информация и бюджет сравнения. Альтернативу и её параметры следует зафиксировать по train/validation до её тестовой оценки, затем сравнить с baseline на том же сохранённом тесте. Если результаты теста используются для итеративного выбора моделей, он становится development-набором; для независимого вывода тогда нужен новый holdout.', '',
        '## Воспроизведение и файлы', '',
        'Окружение: Python ' + run['python'] + ', точные версии в `requirements.lock`; CPU. Данные и модели исключены из Git, их контрольные суммы и пути перечислены в манифестах.', '',
        'Проверка готового локального результата без обучения:', '', '```bash',
        'python main.py verify-test', 'python main.py recommend --user-id 17 --stage test --k 10',
        'python -m scripts.build_report', 'python -m scripts.build_notebook --execute', '```', '',
        'Полное обучение заново выполняется в отдельном каталоге результатов, чтобы сохранить опубликованный тест:', '',
        '```bash', 'python -m scripts.reproduce --name reproduction --full', '```', '',
        'Это повторит валидационную сетку и оба финальных обучения; повторное чтение того же теста проверяет воспроизводимость и не создаёт новый независимый эксперимент. Уже имеющиеся локальные модели повторно не обучаются при обычном `test`.', '',
        '[Выполненный ноутбук](notebooks/kion_baseline.ipynb) использует реальные сохранённые модели и данные: проверяет артефакты и прогнозы, показывает признаки, кандидатов, подбор и итоговую оценку. Все ячейки выполнены; повторное дорогое обучение в нём не запускается автоматически.', '',
        f"Финальная модель: `{run['models']['catboost']['path']}`, SHA-256 `{run['models']['catboost']['model_sha256']}`.",
        f"Код финального запуска: `{run['code_fingerprint']}`. Протокол: `{run['final_protocol_sha256']}`.",
        f"Исходники выполнения: `results/final_execution_sources.json`, Git-коммит обучения `{source.get('git_commit', 'указан в манифесте исходников')}`; локальная копия — `{source.get('snapshot_directory', 'не создавалась')}`.",
        'Основные доказательства: `results/frozen_validation.json`, `results/final_protocol.json`, `results/test_manifest.json`, `results/test_confidence_intervals.json`, `results/test_reload_verification.json`.',
    ]
    from scripts.build_feature_ablation_report import section
    position = lines.index('## Итоговый тест')
    lines[position:position] = section(cfg)
    destination = cfg.root / 'REPORT.md'
    destination.write_text('\n'.join(lines) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config.toml')
    print(build(Config.load(parser.parse_args().config)))
