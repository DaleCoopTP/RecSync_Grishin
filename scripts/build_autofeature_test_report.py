"""Build the commit-25 test record from immutable saved predictions; never fit."""
from pathlib import Path
import argparse
import json

from kion_autofeatures.runtime import configure
configure(2)

import pandas as pd

from kion_autofeatures.config import ResearchConfig
from kion_autofeatures.io import read_json, verify_files, write_json


def audit_comparability(root, run_name, record, frozen):
    directory = root / "results" / "autofeatures" / run_name
    artifact = root / "artifacts" / "autofeatures" / run_name / "test"
    expected = set(frozen["final_variants"]) | {frozen["reference"]}
    if set(record["models"]) != expected or not record["test_evaluated"]:
        raise ValueError("Final evaluation differs from the frozen comparison")
    if record["identity"]["frozen_seal"] != frozen["seal"]:
        raise ValueError("Final results belong to a different frozen method")
    verify_files(directory, record["files"])
    inputs = read_json(directory / "inputs.json")
    allowed = inputs["baseline_config"]["train_cutoffs"] + [inputs["baseline_config"]["validation_cutoff"]]
    reference_pairs = pd.read_parquet(artifact / "full" / "predictions.parquet", columns=["user_id", "item_id"])
    reference_users = pd.read_parquet(artifact / "full" / "per_user.parquet")
    original = read_json(root / inputs["baseline_config"]["results_dir"] / "test_manifest.json")
    original_users = pd.read_parquet(root / original["artifact_directory"] / "catboost_per_user.parquet")
    pd.testing.assert_frame_equal(reference_users, original_users, check_exact=False, rtol=0, atol=1e-10)
    fields = ["target_count", "candidate_count", "candidate_recall@200"]
    parameters = {**frozen["parameters"], "thread_count": frozen["identity"]["research"]["threads"]}
    refits = {}
    for variant in sorted(expected):
        pairs = pd.read_parquet(artifact / variant / "predictions.parquet", columns=["user_id", "item_id"])
        users = pd.read_parquet(artifact / variant / "per_user.parquet")
        pd.testing.assert_frame_equal(pairs, reference_pairs)
        pd.testing.assert_frame_equal(users[fields], reference_users[fields])
        if variant in frozen["final_variants"]:
            fit = read_json(artifact / variant / "fit.json")
            if fit["identity"]["cutoffs"] != allowed or fit["identity"]["parameters"] != parameters:
                raise ValueError("A refit changed temporal cutoffs or fixed parameters")
            if fit["early_stopping"] or fit["test_labels_used"]:
                raise ValueError("A refit used final targets or early stopping")
            if fit["train_rows"] != 2940779 or fit["train_queries"] != 14704:
                raise ValueError("A refit changed the original final-training queries")
            refits[variant] = {"rows": fit["train_rows"], "queries": fit["train_queries"],
                               "features": len(fit["identity"]["columns"]),
                               "parameters": fit["identity"]["parameters"],
                               "model_sha256": fit["files"]["model.cbm"],
                               "test_labels_used": False, "early_stopping": False}
    result = {"variants": sorted(expected), "evaluation_pairs": len(reference_pairs),
              "evaluation_users": len(reference_users), "identical_ordered_pairs": True,
              "identical_users_and_targets": True, "identical_candidate_recall": True,
              "original_full_baseline_per_user_reproduced": True,
              "retrieval_zero_hit_users": int((reference_users["candidate_recall@200"] == 0).sum()),
              "training_cutoffs": allowed, "test_cutoff": inputs["baseline_config"]["test_cutoff"],
              "refits": refits, "frozen_seal": frozen["seal"], "test_evaluated": True}
    write_json(directory / "test_comparability.json", result)
    return result


def build(root, run_name="main"):
    ResearchConfig(run_name=run_name)
    directory = root / "results" / "autofeatures" / run_name
    artifact = root / "artifacts" / "autofeatures" / run_name
    record = read_json(directory / "test.json")
    frozen = read_json(directory / "frozen.json")
    intervals = read_json(directory / "test_intervals.json")
    table = pd.read_csv(directory / "test_results.csv")
    checked = audit_comparability(root, run_name, record, frozen)
    names = {"source": "S: сигналы источников", "source_raw": "S + R: сигналы и метаданные",
             "auto": "S + R + A: автоматический отбор",
             "random": "S + R + A_random: случайный контроль", "full": "S + R + M: полный baseline"}
    lines = ["# Коммит 25: итоговая тестовая оценка", "",
             "Метод зафиксирован до тестовой оценки новых моделей. Четыре модели",
             "переобучены на одинаковых четырёх исторических срезах; полный baseline",
             "загружен из первоначальной финальной оценки, его метрики воспроизведены.",
             "После просмотра теста признаки и параметры не изменялись.", "",
             "## Протокол и сопоставимость", "",
             f"- Обучение: 2940779 пар, 14704 запросов; срезы {', '.join(checked['training_cutoffs'])}.",
             f"- Тест: {checked['test_cutoff']}, горизонт 7 дней; {checked['evaluation_users']} пользователей",
             f"  и {checked['evaluation_pairs']} упорядоченных пар кандидатов.",
             f"- У {checked['retrieval_zero_hit_users']} пользователей в пуле нет позитивов; они остаются в метриках.",
             f"- Автоматический набор: {len(frozen['selection'])} агрегатов; случайный набор того же размера.",
             "- Depth=8, iterations=299, learning_rate=0.05, L2=3, seed=42, два потока.",
             "- Все четыре переобучения завершены до построения тестовых целей.",
             "- Нового подбора, eval_set и early stopping на финальном этапе нет.", "",
             "## Качество", "",
             "| Вариант | Признаков | NDCG@10 | Recall@10 | MAP@10 |",
             "|---|---:|---:|---:|---:|"]
    for variant in ("source", "source_raw", "full", "auto", "random"):
        row = table.loc[table.variant == variant].iloc[0]
        features = len(record["models"][variant]["identity"]["columns"])
        lines.append(f"| {names[variant]} | {features} | {row['ndcg@10']:.8f} | {row['recall@10']:.8f} | {row['map@10']:.8f} |")
    lines += ["", f"Общий macro candidate Recall@200 = {table['candidate_recall@200'].iloc[0]:.8f}.",
              "Знаменатели метрик используют полные будущие цели, включая не попавшие в кандидаты фильмы.", "",
              "## Парные различия", ""]
    contrasts = {"auto − full (первичное)": intervals["primary_auto_minus_full"],
                 **{"auto − " + variant: values for variant, values in intervals["diagnostic_auto_minus"].items()}}
    for label, values in contrasts.items():
        delta = values["ndcg@10"]
        lines.append(f"- {label}: ΔNDCG={delta['mean']:+.8f}; 95% CI [{delta['lower']:+.8f}; {delta['upper']:+.8f}].")
    lines += ["", "Percentile bootstrap по одним и тем же пользователям: 1000 повторов, seed=42.",
              "Первичное сравнение задано до теста; остальные диагностические, без поправки",
              "за множественные сравнения. Интервалы не учитывают переобучение или смену недели.", ""]
    primary = intervals["primary_auto_minus_full"]["ndcg@10"]
    if primary["lower"] > 0:
        lines += ["Первичный интервал выше нуля: на этой тестовой неделе автоматический вариант",
                  "имеет более высокий NDCG@10 при данных фиксированных моделях.", ""]
    elif primary["upper"] < 0:
        lines += ["Первичный интервал ниже нуля: на этой тестовой неделе автоматический вариант",
                  "имеет более низкий NDCG@10 при данных фиксированных моделях.", ""]
    else:
        lines += ["Первичный интервал включает ноль. Превосходство автоматического варианта",
                  "не подтверждено. Интервал допускает как ухудшение, так и улучшение;",
                  "равенство качества из этого результата не следует.", ""]
    random_delta = intervals["diagnostic_auto_minus"]["random"]["ndcg@10"]
    if random_delta["upper"] < 0:
        lines += ["На этой неделе выбранный набор уступил одному заранее фиксированному",
                  "случайному набору по NDCG@10: диагностический интервал auto − random",
                  "ниже нуля. Это результат одного случайного контроля и одного периода,",
                  "без поправки за множественные сравнения; он не доказывает превосходство",
                  "случайного поиска в целом. Основной метод после теста не меняется.", ""]
    elif random_delta["lower"] > 0:
        lines += ["На этой неделе выбранный набор превзошёл один фиксированный случайный",
                  "набор по NDCG@10; это диагностическое сравнение без поправки за",
                  "множественные сравнения, а не вывод обо всём случайном поиске.", ""]
    else:
        lines += ["Убедительного преимущества отбора над одним фиксированным случайным",
                  "набором на этой неделе не выявлено. Основной метод после теста не меняется.", ""]
    lines += ["## Затраты финального этапа", "",
              "| Вариант | Pool обучения, с | Fit, с | Predict, с | Размер модели, МиБ |",
              "|---|---:|---:|---:|---:|"]
    for variant in ("source", "source_raw", "auto", "random", "full"):
        model = record["models"][variant]
        pool = f"{model['training_pool_seconds']:.3f}" if "training_pool_seconds" in model else "—"
        fit = f"{model['fit_seconds']:.2f}" if model.get("fit_seconds") is not None else "переиспользована"
        lines.append(f"| {names[variant]} | {pool} | {fit} | {model['predict_seconds_median']:.3f} | {model['model_bytes']/1024**2:.3f} |")
    attempts = []
    audit_path = directory / "test_verification.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        attempts = [audit["resources"]["test"], audit["resources"]["cached_test"]]
    else:
        for path in sorted((artifact / "resources").glob("test_*.json")):
            value = read_json(path)
            if value["command"] == "test":
                attempts.append(value)
    if attempts:
        cold = attempts[0]
        lines += ["", f"Первый запуск финального этапа: {cold['seconds']:.2f} с,",
                  f"peak tree RSS {cold['peak_tree_rss_mb']:.1f} МиБ; "
                  f"код завершения {cold['returncode']}, превышение лимита RSS: "
                  f"{'да' if cold['memory_limit_exceeded'] else 'нет'}."]
        if len(attempts) > 1:
            lines += [f"Повторный вызов с готовыми результатами: {attempts[-1]['seconds']:.2f} с.",
                      "Он проверяет сохранённые артефакты и не является новым обучением или независимым тестом."]
    lines += ["", "Predict — медиана трёх повторов после прогрева на готовых признаках.",
              "Pool и fit разделены. Это пакетный offline-замер; retrieval и построение",
              "признаков не входят в predict. Варианты обучались последовательно в одном",
              "рабочем процессе этапа, поэтому RSS зависит от состояния аллокатора и не",
              "является независимым сравнением памяти моделей.", "",
              "## Артефакты и ограничения", "",
              "- [test_results.csv](test_results.csv): метрики и затраты.",
              "- [test_intervals.json](test_intervals.json): парные интервалы всех трёх метрик.",
              "- [test_segments.csv](test_segments.csv): срезы по длине истории и попаданию целей в кандидаты.",
              "- [test_comparability.json](test_comparability.json): совпадение пар, пользователей и целей.",
              "- [test_pre_evaluation_checkpoint.json](test_pre_evaluation_checkpoint.json): контрольная точка до окончания четвёртого переобучения.",
              "- [test_verification.json](test_verification.json): загрузка моделей, повторный запуск и сохранность входов.",
              "- [frozen.json](frozen.json): зафиксированный до оценки метод.", "",
              "Тестовые результаты baseline были известны ранее; это ограничение явно",
              "зафиксировано в протоколе. Параметры новых моделей выбирались без их тестовой оценки.",
              "Исследование ограничено одной неделей, одним seed, last-watch форматом KION,",
              "статическими метаданными и warm-active пользователями. Сигналы источников и",
              "персональные кандидаты остаются инженерными компонентами. Случайный контроль один.",
              "Результат не доказывает онлайн-эффект или универсальное превосходство метода.",
              "Полный исследовательский отчёт и выполненный ноутбук относятся к коммиту 26."]
    (directory / "TEST.md").write_text("\n".join(lines) + "\n")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="main")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.name), indent=2, ensure_ascii=False))

