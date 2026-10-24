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
        conclusion = 'На этой валидации дополнительные ручные признаки улучшают модель поверх сигналов источников.'
    elif delta['upper'] < 0:
        conclusion = 'На этой валидации добавление ручных признаков ухудшает результат при фиксированной конфигурации.'
    else:
        conclusion = 'Устойчивый дополнительный выигрыш ручных признаков поверх сигналов источников этой диагностикой не подтверждён.'
    if groups_delta['upper'] < 0:
        family_conclusion = ('Компактная комбинация сигналов источников превосходит вариант только с ручными '
                             'признаками на этой валидации. Большего числа столбцов недостаточно для лучшего качества; '
                             'ручные агрегаты полезны прежде всего как дополнение к сигналам в полной модели.')
    elif groups_delta['lower'] > 0:
        family_conclusion = 'Вариант только с ручными признаками превосходит компактную комбинацию сигналов на этой валидации.'
    else:
        family_conclusion = 'Различие двух отдельных семейств этим описательным сравнением устойчиво не подтверждено.'
    params = plan['parameters']
    train_rows = sum(s['rows'] for s in plan['training_snapshots'])
    train_queries = sum(s['queries'] for s in plan['training_snapshots'])
    return [
        '## Дополнительная абляция групп признаков на валидации', '',
        f"Диагностика запланирована {plan['created_at_utc']} после завершения исходного baseline. "
        'Обучены ровно два новых варианта: только сигналы источников и только ручные признаки. '
        'Полная модель загружена из первоначальной валидации, её прогнозы воспроизведены. '
        'Новые варианты на тесте не оценивались; исходные модели, тестовые результаты и протокол сохранены побайтно.', '',
        f"Во всех вариантах одни и те же три обучающих среза, {train_rows:,} пар, {train_queries:,} групп и "
        f"{plan['full_model']['metrics']['users']:,} валидационных пользователей. "
        f"Параметры полной модели фиксированы: depth={params['depth']}, l2={params['l2_leaf_reg']}, "
        f"learning_rate={params['learning_rate']}, seed={params['random_seed']}, {params['iterations']} деревьев, "
        f"{params['thread_count']} потока. Новые модели обучены без eval_set и early stopping; отдельного подбора для групп нет.", '',
        f"**Сигналы источников ({len(plan['feature_groups']['source_only'])}):** все `cand_*` (баллы, ранги, присутствие) и `rrf_score`. "
        f"**Ручные признаки ({len(plan['feature_groups']['manual_only'])}):** все `u_*`, `i_*`, `pair_*`. "
        'Семейства разделяют столбцы, но частично дублируют информацию: например, популярность доступна и как балл источника, '
        'и как счётчик фильма. Сами сигналы источников тоже спроектированы человеком. Это сравнение представлений признаков, '
        'а не ручного и автоматически обученного представления.', '',
        markdown_table(table, {'variant': 'Вариант', 'features': 'Признаков', 'ndcg@10': 'NDCG@10',
                              'recall@10': 'Recall@10', 'map@10': 'MAP@10'}, fmt), '',
        '![Абляция групп признаков на валидации](results/feature_ablation_metrics.png)', '',
        f"Полная модель минус сигналы источников: ΔNDCG={delta['mean']:+.5f} ({relative:+.2f}%), "
        f"парный 95% интервал [{delta['lower']:+.5f}; {delta['upper']:+.5f}]. {conclusion}",
        f"Полная модель минус ручные признаки: ΔNDCG={manual_delta['mean']:+.5f}, "
        f"парный 95% интервал [{manual_delta['lower']:+.5f}; {manual_delta['upper']:+.5f}]. "
        'Это совместный эффект добавленной группы при данных параметрах, а не оценка каждого признака.', '',
        family_conclusion,
        f"Только ручные признаки минус только сигналы: ΔNDCG={groups_delta['mean']:+.5f}, "
        f"парный 95% интервал [{groups_delta['lower']:+.5f}; {groups_delta['upper']:+.5f}].", '',
        f"Bootstrap: {plan['bootstrap_samples']} повторов по одним и тем же пользователям, seed={plan['seed']}. "
        'Валидация ранее использовалась для выбора полной модели; интервалы описательные, без коррекции множественных сравнений, '
        'не учитывают переобучение и смену периода. Они не заменяют независимую тестовую проверку новых вариантов.', '',
        markdown_table(table, {'variant': 'Вариант', 'fit_seconds': 'Fit, с',
                              'fit_process_peak_rss_mb': 'RSS обучения, МиБ',
                              'predict_seconds_median': 'Predict, с',
                              'inference_process_peak_rss_mb': 'RSS предсказания, МиБ',
                              'model_mib': 'Модель, МиБ'}, fmt), '',
        'Fit — время только вызова обучения без построения Pool. RSS обучения — пик отдельного процесса '
        'с чтением обучающих признаков, построением Pool и обучением. Predict — медиана трёх вызовов на всей '
        'валидации после прогрева, без построения Pool. RSS предсказания для каждого варианта измерен в отдельном '
        'процессе: чтение готовых признаков, загрузка модели, Pool и predict. Это пакетный offline-замер, не латентность сервиса.',
        f"Полная модель заново не обучалась: исходный этап fit + predict + оценка занимал "
        f"{plan['full_model']['historical_fit_predict_evaluate_seconds']:.1f} с, обучено "
        f"{plan['full_model']['original_trained_iterations']} деревьев и сохранено {params['iterations']}. "
        'Отдельные время fit и RSS именно её обучения тогда не измерялись, поэтому эти клетки оставлены пустыми '
        'и отношение скоростей обучения к ней не рассчитывается.',
        'Во всех вариантах используется один персональный пул и готовый кэш. Стоимость извлечения кандидатов '
        'и расчёта исходных признаков в эти замеры не входит. Вариант без сигналов источников по-прежнему получает '
        'кандидатов от ItemKNN и жанрового источника; экономия всей системы и отключение персонализации не проверялись.', '',
        '**Отправная точка для второй части:** заменить или автоматически построить семейства `u_*`, `i_*`, `pair_*`, '
        'сохранив одинаковые кандидаты и доступную историю. Сравнивать альтернативу следует и с полной моделью, '
        'и с компактной комбинацией сигналов источников. Эти результаты не меняют зафиксированный baseline '
        'и не используются для подбора по тесту.', '',
        'Воспроизведение: `python -m scripts.feature_ablation`. Завершённые модели и результаты проверяются '
        'по контрольным суммам и используются повторно. Протокол и доказательства: '
        '`results/feature_ablation_protocol.json`, `results/feature_ablation_results.csv`, '
        '`results/feature_ablation_intervals.json`, `results/feature_ablation_manifest.json`.', '',
    ]
