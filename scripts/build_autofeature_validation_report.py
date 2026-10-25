"""Build the commit-24 validation record from saved outputs, without fitting."""
from collections import Counter
import json
from pathlib import Path

from kion_autofeatures.runtime import configure
configure(2)

import pandas as pd

from kion_autofeatures.grammar import FeatureSpec
from kion_autofeatures.io import read_json, write_json


def audit_comparability(root, directory, validation):
    artifact = root / "artifacts" / "autofeatures" / directory.name / "validation"
    pairs = pd.read_parquet(artifact / "full" / "predictions.parquet", columns=["user_id", "item_id"])
    reference = pd.read_parquet(artifact / "full" / "per_user.parquet")
    fields = ["target_count", "candidate_count", "candidate_recall@200"]
    for name, record in validation["models"].items():
        actual = pd.read_parquet(artifact / name / "predictions.parquet", columns=["user_id", "item_id"])
        pd.testing.assert_frame_equal(actual, pairs)
        per_user = pd.read_parquet(artifact / name / "per_user.parquet")
        pd.testing.assert_frame_equal(per_user[fields], reference[fields])
        if not record.get("training_reused", False):
            if record["train_rows"] != 2223979 or record["train_queries"] != 11120:
                raise ValueError("New variant does not use the original training queries")
    result = {"variants": sorted(validation["models"]), "identical_ordered_pairs": True,
              "identical_evaluation_users_and_targets": True, "identical_candidate_recall": True,
              "training_rows_per_new_variant": 2223979, "training_queries_per_new_variant": 11120,
              "evaluation_pairs": len(pairs), "evaluation_users": len(reference),
              "test_evaluated": False}
    write_json(directory / "comparability.json", result)
    return result


def manual_equivalents():
    result = {}
    for root, prefix in (("user", "u"), ("item", "i")):
        for window in (0, 7, 30, 90):
            result[FeatureSpec(root, "count", window).name] = f"{prefix}_count_{window or 'all'}"
        result[FeatureSpec(root, "recency").name] = f"{prefix}_recency"
        result[FeatureSpec(root, "mean_log_dur").name] = f"{prefix}_total_dur_mean"
        result[FeatureSpec(root, "mean_pct").name] = f"{prefix}_watched_pct_mean"
        result[FeatureSpec(root, "trend", 7, 30).name] = f"{prefix}_trend_7_30"
    for category, prefix in (("genres", "genre"), ("directors", "directors"), ("content_type", "content_type")):
        for window in (0, 7, 30, 90):
            for op in ("count", "share"):
                result[FeatureSpec("user_" + category, op, window).name] = f"pair_{prefix}_{op}_{window or 'all'}"
        result[FeatureSpec("user_" + category, "recency").name] = f"pair_{prefix}_recency"
    return result


def build(root, name="main"):
    from kion_autofeatures.config import ResearchConfig
    ResearchConfig(run_name=name)
    run_name = name
    directory = root / "results" / "autofeatures" / run_name
    selection = read_json(directory / "selection.json")
    validation = read_json(directory / "validation.json")
    intervals = read_json(directory / "validation_intervals.json")
    table = pd.read_csv(directory / "validation_results.csv")
    audit_comparability(root, directory, validation)
    equivalents = manual_equivalents()
    selected = selection["selected"]
    rows = []
    for group, names in (("selected", selected), ("random", selection["random"])):
        for name in names:
            spec = selection["feature_definitions"][name]
            rows.append({"set": group, "feature": name, "path": spec["path"],
                         "operation": spec["operation"], "window": spec["window"] or "all",
                         "long_window": spec["long_window"] or "",
                         "manual_semantic_equivalent": equivalents.get(name, "")})
    pd.DataFrame(rows).to_csv(directory / "selected_features.csv", index=False)
    descriptions = [selection["feature_definitions"][name] for name in selected]
    summary = {"generated": len(selection["feature_definitions"]), "eligible": len(selection["eligible"]),
               "rejected": len(selection["rejected"]), "selected": len(selected),
               "paths": dict(Counter(s["path"] for s in descriptions)),
               "operations": dict(Counter(s["operation"] for s in descriptions)),
               "windows": dict(Counter(f"{s['window']}/{s['long_window']}" if s["operation"] == "trend"
                                       else str(s["window"] or "all") for s in descriptions)),
               "semantic_overlap_with_manual": {name: equivalents[name] for name in selected if name in equivalents},
               "note": "Semantic correspondence only; missing values, numeric precision and means can differ"}
    importance = pd.read_csv(root / "artifacts" / "autofeatures" / run_name / "validation" / "auto" / "importance.csv")
    importance["family"] = importance.feature.map(
        lambda name: "A" if name.startswith("auto__") else "S" if name.startswith("cand_") or name == "rrf_score" else "R")
    importance.sort_values("importance", ascending=False).to_csv(directory / "validation_feature_importance.csv", index=False)
    summary["validation_importance_by_family"] = importance.groupby("family").importance.sum().to_dict()
    write_json(directory / "selection_summary.json", summary)
    display = {"source": "S: сигналы источников", "source_raw": "S + R: сигналы и метаданные",
               "manual": "R + M: ручные признаки", "full": "S + R + M: полный baseline",
               "auto_no_source": "R + A: автоматические признаки", "auto": "S + R + A: основной автоматический",
               "random": "S + R + A_random: случайный контроль"}
    lines = ["# Коммит 24: основной поиск и внешняя валидация", "",
             "Тестовая оценка нового метода ещё не выполнена. После этой валидации метод фиксируется;",
             "пространство поиска и параметры не меняются по её результатам.", "",
             "## Метод и выбор признаков", "",
             f"Грамматика: 504 признака; после удаления констант/дубликатов осталось {summary['eligible']}.",
             f"Выбрано {len(selected)} признаков по внутренней валидации 2 августа, до внешней валидации.",
             "Основное обучение использует исходные три среза и 2223979 пар / 11120 запросов;",
             "внешняя валидация — исходные 2170393 пары и 10852 пользователя на 9 августа.", "",
             "| Префикс | Внутренний NDCG@10 |", "|---|---:|"]
    lines += [f"| {r['size']} | {r['metrics']['ndcg@10']:.8f} |" for r in selection["prefixes"]]
    lines += ["", "Если отставание от лучшего не превышает 0,0001, выбирается меньший префикс.",
              "Затем все четыре новых варианта обучены с фиксированными depth=8, L2=3,",
              "lr=0.05, seed=42 и 299 деревьями, без eval_set и early stopping.", "",
              "Вариант R+A использует тот же A, отобранный в присутствии S+R; отдельного",
              "поиска специально для модели без сигналов источников не проводилось.", "",
              "## Качество на внешней валидации", "",
              "| Вариант | NDCG@10 | Recall@10 | MAP@10 |",
              "|---|---:|---:|---:|"]
    for name in ("source", "source_raw", "manual", "full", "auto_no_source", "auto", "random"):
        row = table.loc[table.variant == name].iloc[0]
        lines.append(f"| {display[name]} | {row['ndcg@10']:.8f} | {row['recall@10']:.8f} | {row['map@10']:.8f} |")
    lines += ["", "## Парные различия", ""]
    contrasts = {"auto − full": intervals["primary_auto_minus_full"],
                 **{"auto − " + name: value for name, value in intervals["diagnostic_auto_minus"].items()}}
    for name, values in contrasts.items():
        value = values["ndcg@10"]
        lines.append(f"- {name}: ΔNDCG={value['mean']:+.8f}; 95% CI [{value['lower']:+.8f}; {value['upper']:+.8f}].")
    lines += ["", "Интервалы описательные: валидация ранее использовалась для выбора baseline;",
              "пользователи ресэмплируются при фиксированных моделях. Нет поправки за",
              "множественные сравнения, неопределённость переобучения и смену периода.",
              "Отсутствие убедительного различия не доказывает равенство качества.", ""]
    primary = intervals["primary_auto_minus_full"]["ndcg@10"]
    random_contrast = intervals["diagnostic_auto_minus"]["random"]["ndcg@10"]
    if primary["lower"] <= 0 <= primary["upper"]:
        lines += ["Превосходство автоматического варианта над полным baseline на этой",
                  "валидации не подтверждено. Интервал допускает как ухудшение, так и улучшение.", ""]
    if random_contrast["lower"] <= 0 <= random_contrast["upper"]:
        lines += ["Специальный отбор также не показал убедительного преимущества над одним",
                  "зафиксированным случайным набором. По этому сравнению нельзя приписывать",
                  "близость точечных оценок к baseline именно ранжированию признаков по важности;",
                  "эффекты синтеза агрегатов и специального отбора нужно различать.", ""]
    lines += [
              "## Какие признаки выбраны", "",
              f"Пути: {json.dumps(summary['paths'], ensure_ascii=False, sort_keys=True)}.",
              f"Операции: {json.dumps(summary['operations'], ensure_ascii=False, sort_keys=True)}.",
              f"Окна: {json.dumps(summary['windows'], ensure_ascii=False, sort_keys=True)}.",
              f"Содержательные аналоги ручных признаков: {len(summary['semantic_overlap_with_manual'])} из {len(selected)}.",
              "Это соответствие формул, а не обещание побитового равенства: отличаются",
              "обработка пропусков и местами численная точность.",
              "Полный выбранный и случайный наборы: [selected_features.csv](selected_features.csv).", "",
              "Важность PredictionValuesChange основной модели по группам: "
              + "; ".join(f"{group}={summary['validation_importance_by_family'][group]:.2f}%"
                          for group in ("S", "R", "A")) + ".",
              "Это описательная важность обученной модели, не причинный эффект и не",
              "доказательство независимой полезности каждой коррелирующей колонки.",
              "Подробности: [validation_feature_importance.csv](validation_feature_importance.csv).", "",
              "## Вычислительные затраты", "",
              "| Вариант | Fit, с | Predict, с | RSS процесса, МиБ |",
              "|---|---:|---:|---:|"]
    for name in ("source_raw", "auto_no_source", "auto", "random", "full", "source", "manual"):
        row = table.loc[table.variant == name].iloc[0]
        fit = f"{row.fit_seconds:.2f}" if pd.notna(row.fit_seconds) else "модель переиспользована"
        lines.append(f"| {display[name]} | {fit} | {row.predict_seconds_median:.3f} | {row.process_peak_rss_mb:.1f} |")
    lines += ["", "Генерация общего объединения выбранных и случайных признаков:", "",
              "| Срез и роль | Пар | Автопризнаков | Расчёт кэша, с | Всё обращение, с | Кэш, МиБ |",
              "|---|---:|---:|---:|---:|---:|"]
    for generated in validation["generation"]:
        cache = generated["cache"]
        role = "train" if generated["training"] else "validation"
        lines.append(f"| {generated['cutoff']} {role} | {cache['rows']} | "
                     f"{len(cache['identity']['features'])} | {cache['seconds']:.2f} | "
                     f"{generated['request_seconds']:.2f} | {cache['bytes'] / 1024**2:.2f} |")
    lines += ["", "Всё обращение включает подготовку исторического вычислителя; расчёт кэша —",
              "его transform и запись. Это стоимость общего кэша двух наборов, а не",
              "раздельный замер генерации каждого варианта или всего сервиса."]
    resources = []
    for path in sorted((root / "artifacts" / "autofeatures" / run_name / "resources").glob("*.json")):
        value = read_json(path)
        if value["command"] in ("search", "validate"):
            resources.append(value)
    for value in resources:
        lines.append(f"\nЭтап {value['command']}: {value['seconds']:.2f} с, peak tree RSS {value['peak_tree_rss_mb']:.1f} МиБ; "
                     f"код завершения {value['returncode']}.")
    lines += ["", "Два потока, процессы последовательные, лимит суммы RSS — 6 GiB.",
              "Fit измеряет вызов обучения; Pool записан отдельно в model receipts.",
              "Predict — медиана трёх повторов после прогрева. RSS включает загруженные",
              "данные и Pool; это не размер модели. Полный baseline и два готовых",
              "контроля заново не обучались. Стоимость общего retrieval не входит",
              "в fit/predict и не объявляется устранённой.",
              "Варианты измерялись последовательно в одном рабочем процессе этапа;",
              "RSS зависит и от состояния аллокатора после предыдущих вариантов.",
              "Это не независимые замеры памяти моделей: экономия памяти относительно",
              "baseline по этим строкам не заявляется.", "",
              "## Фиксация и ограничения", "",
              "S — 13 инженерных сигналов источников; R — 15 статических колонок;",
              "M — 54 ручных агрегата; A — выбранные автоматические агрегаты.",
              "Персональные кандидаты одинаковы. Генератор не получает labels;",
              "срезы и все доступные пользовательские истории сохранены.",
              "Выбор признаков идёт по двум более ранним обучающим срезам и внутренней валидации.",
              "Результаты основного теста baseline уже известны; новые модели будут",
              "оценены только после фиксации в отдельном коммите 25.",
              "Состав финального сравнения фиксирован независимо от знака валидационного эффекта.",
              "Данные имеют last-watch формат; метаданные считаются статическими.",
              "Одна неделя, один seed и один случайный набор ограничивают выводы.",
              "Финальное состояние и контрольные суммы — [frozen.json](frozen.json)."]
    (directory / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="main")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.name), indent=2, ensure_ascii=False))

