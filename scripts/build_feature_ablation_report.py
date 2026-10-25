"""Validation feature-family comparison for the main report."""
import os

from scripts.feature_ablation import LABELS, read_json, verify_plan


def section(cfg):
    import numpy as np
    import pandas as pd
    from kion.io import sha256
    from scripts.build_report import markdown_table
    results = cfg.path('results_dir')
    if not (results / 'feature_ablation_manifest.json').exists():
        return []
    plan = read_json(results / 'feature_ablation_protocol.json')
    manifest = read_json(results / 'feature_ablation_manifest.json')
    from scripts.feature_ablation import digest
    verify_plan(cfg, plan)
    if manifest['protocol_sha256'] != digest(plan) or manifest['test_evaluated']:
        raise ValueError('Invalid feature ablation result identity')
    for name, checksum in manifest['result_files'].items():
        if sha256(results / name) != checksum:
            raise ValueError('Feature ablation result checksum mismatch')
    frame = pd.read_csv(results / 'feature_ablation_results.csv')
    intervals = read_json(results / 'feature_ablation_intervals.json')
    table = frame.copy()
    table['variant'] = table.variant.map(LABELS)
    fmt = {metric: lambda x: f'{x:.5f}' for metric in ['ndcg@10', 'recall@10', 'map@10']}
    fmt.update({key: lambda x: 'не измерено' if pd.isna(x) else f'{x:.1f}'
                for key in ['fit_seconds', 'fit_process_peak_rss_mb', 'inference_process_peak_rss_mb']})
    fmt['predict_seconds_median'] = lambda x: f'{x:.3f}'
    fmt['model_mib'] = lambda x: f'{x:.2f}'
    os.environ.setdefault('MPLCONFIGDIR', str(cfg.path('artifact_dir') / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered = frame.sort_values('ndcg@10')
    values = ordered['ndcg@10'].to_numpy()
    errors = np.vstack([np.maximum(0, values - ordered['ndcg@10_lower']),
                        np.maximum(0, ordered['ndcg@10_upper'] - values)])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.barh([LABELS[v] for v in ordered.variant], values, xerr=errors, capsize=4,
            color=['#2463a6' if v == 'full' else '#9aabbc' for v in ordered.variant])
    for pos, row in enumerate(ordered.to_dict('records')):
        ax.text(row['ndcg@10_upper'] + .002, pos, f"{row['ndcg@10']:.5f}", va='center')
    ax.set_xlim(0, float(ordered['ndcg@10_upper'].max()) * 1.23)
    ax.set_xlabel('Macro NDCG@10; описательные 95% bootstrap-интервалы')
    ax.set_title('Группы признаков: одна валидация и одинаковые кандидаты\nДополнительная диагностика; параметры фиксированы')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(results / 'feature_ablation_metrics.png', dpi=160)
    plt.close(fig)
    indexed = frame.set_index('variant')
    delta = intervals['paired']['full_minus_source_only']['ndcg@10']
    relative = 100 * (indexed.loc['full', 'ndcg@10'] / indexed.loc['source_only', 'ndcg@10'] - 1)
    manual_delta = intervals['paired']['full_minus_manual_only']['ndcg@10']
    groups_delta = intervals['paired']['manual_only_minus_source_only']['ndcg@10']
    if delta['lower'] > 0:
        conclusion = 'Ручные признаки улучшают модель поверх сигналов источников.'
    elif delta['upper'] < 0:
        conclusion = 'Ручные признаки ухудшают модель при фиксированных параметрах.'
    else:
        conclusion = 'Устойчивого выигрыша от ручных признаков поверх сигналов источников не видно.'
    if groups_delta['upper'] < 0:
        family_conclusion = ('13 сигналов источников работают лучше 69 ручных признаков: дело не в числе колонок. '
                             'Ручные признаки полезны прежде всего как дополнение к сигналам в полной модели.')
    elif groups_delta['lower'] > 0:
        family_conclusion = 'Модель только на ручных признаках работает лучше модели только на сигналах источников.'
    else:
        family_conclusion = 'Устойчивой разницы между моделями только на сигналах и только на ручных признаках не видно.'
    params = plan['parameters']
    train_rows = sum(s['rows'] for s in plan['training_snapshots'])
    train_queries = sum(s['queries'] for s in plan['training_snapshots'])
    def grouped(value):
        return f'{value:,}'.replace(',', ' ')

    return [
        '## Абляция групп признаков на валидации', '',
        'После фиксации baseline обучены две дополнительные модели: только на сигналах источников и только '
        'на ручных признаках. Полная модель взята из исходной валидации, её прогнозы воспроизведены. '
        'Новые модели на тесте не оценивались; исходные модели, тестовые результаты и протокол не менялись.', '',
        f"Во всех вариантах те же три обучающих среза (пар {grouped(train_rows)}, запросов {grouped(train_queries)}) "
        f"и те же валидационные пользователи ({grouped(plan['full_model']['metrics']['users'])}). Параметры полной модели "
        f"не менялись: depth={params['depth']}, l2={params['l2_leaf_reg']}, learning_rate={params['learning_rate']}, "
        f"seed={params['random_seed']}, {params['iterations']} деревьев, {params['thread_count']} потока, без eval_set "
        'и early stopping.', '',
        f"**Сигналы источников ({len(plan['feature_groups']['source_only'])}):** все `cand_*` (баллы, ранги и флаги "
        f"присутствия четырёх источников кандидатов) и `rrf_score`; в основном отчёте это группа S. "
        f"**Ручные признаки ({len(plan['feature_groups']['manual_only'])}):** все `u_*`, `i_*`, `pair_*`. "
        'Колонки групп не пересекаются, но информация частично дублируется: популярность фильма есть и как балл '
        'источника, и как счётчик. Сигналы источников тоже придуманы вручную, так что здесь сравниваются два '
        'ручных представления, а не ручное и обученное.', '',
        markdown_table(table, {'variant': 'Вариант', 'features': 'Признаков', 'ndcg@10': 'NDCG@10',
                              'recall@10': 'Recall@10', 'map@10': 'MAP@10'}, fmt), '',
        '![Абляция групп признаков на валидации](../results/feature_ablation_metrics.png)', '',
        f"Полная модель против модели на сигналах источников: ΔNDCG = {delta['mean']:+.5f} ({relative:+.2f}%), "
        f"парный 95% CI [{delta['lower']:+.5f}; {delta['upper']:+.5f}]. {conclusion}",
        f"Полная модель против модели на ручных признаках: ΔNDCG = {manual_delta['mean']:+.5f}, "
        f"95% CI [{manual_delta['lower']:+.5f}; {manual_delta['upper']:+.5f}]. "
        'Это совместный эффект добавленной группы, а не вклад отдельных признаков.', '',
        family_conclusion,
        f"Ручные признаки против сигналов источников: ΔNDCG = {groups_delta['mean']:+.5f}, "
        f"95% CI [{groups_delta['lower']:+.5f}; {groups_delta['upper']:+.5f}].", '',
        f"Bootstrap: {plan['bootstrap_samples']} повторов по одним и тем же пользователям, seed = {plan['seed']}. "
        'Эта валидация уже использовалась для выбора полной модели, поэтому интервалы описательные и без поправки '
        'на множественность. Независимой тестовой проверки новых вариантов они не заменяют.', '',
        markdown_table(table, {'variant': 'Вариант', 'fit_seconds': 'Обучение, с',
                              'fit_process_peak_rss_mb': 'Память при обучении, МиБ',
                              'predict_seconds_median': 'Предсказание, с',
                              'inference_process_peak_rss_mb': 'Память при предсказании, МиБ',
                              'model_mib': 'Модель, МиБ'}, fmt), '',
        'Обучение: время самого вызова fit, без построения Pool. Память при обучении: пик отдельного процесса, '
        'который читает признаки, строит Pool и обучает модель. Предсказание: медиана трёх вызовов на всей '
        'валидации после прогрева, без построения Pool. Память при предсказании измерена в отдельном процессе '
        'для каждого варианта: чтение признаков, загрузка модели, Pool и predict. Это пакетный замер, '
        'а не задержка сервиса.',
        f"Полная модель заново не обучалась. Исходный этап fit, predict и оценки занимал "
        f"{plan['full_model']['historical_fit_predict_evaluate_seconds']:.1f} с: обучено "
        f"{plan['full_model']['original_trained_iterations']} деревьев, сохранено {params['iterations']}. "
        'Время и память именно её обучения тогда отдельно не измерялись, поэтому в таблице их нет.',
        'Все варианты используют один персональный пул и готовый кэш признаков; построение кандидатов и признаков '
        'в замеры не входит. Модель без сигналов источников всё равно получает кандидатов от ItemKNN и жанрового '
        'источника, так что экономия на всей системе здесь не оценивалась.', '',
        '**Вывод для автоматических признаков:** заменять или строить автоматически стоит семейства '
        '`u_*`, `i_*`, `pair_*` при тех же кандидатах и истории, а сравнивать и с полной моделью, и с моделью '
        'только на сигналах источников.', '',
        'Воспроизведение: `python -m scripts.feature_ablation`. Готовые модели и результаты проверяются по '
        'контрольным суммам и используются повторно. Протокол и результаты: '
        '`results/feature_ablation_protocol.json`, `results/feature_ablation_results.csv`, '
        '`results/feature_ablation_intervals.json`, `results/feature_ablation_manifest.json`.', '',
    ]
