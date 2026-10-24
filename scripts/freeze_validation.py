"""Freeze completed validation results without accessing test targets.

Run from the project root: python -m scripts.freeze_validation
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import pandas as pd
from catboost import CatBoostRanker
from kion.config import Config
from kion.io import sha256, write_json
from kion.pipeline import code_fingerprint


def freeze(cfg):
    results = cfg.path("results_dir")
    run = json.loads((results / "validation_manifest.json").read_text())
    if run["split"] != "validation" or run["test_evaluated"] or run["cutoff"] != cfg.validation_cutoff:
        raise ValueError("Only the planned, untouched validation experiment may be frozen")
    if run["config_fingerprint"] != cfg.fingerprint() or run["code_fingerprint"] != code_fingerprint():
        raise ValueError("Configuration or pipeline source changed after the experiment")
    model_path = cfg.root / run["model_artifact"]
    if sha256(model_path) != run["model_sha256"]:
        raise ValueError("Selected model checksum mismatch")
    model = json.loads((model_path.parent / "model_manifest.json").read_text())
    if ({(r["depth"], r["l2_leaf_reg"]) for r in model["grid"]} != {(d, l2) for d in cfg.depths for l2 in cfg.l2_values}
            or len(model["grid"]) != len(cfg.depths) * len(cfg.l2_values)):
        raise ValueError("Incomplete hyperparameter grid")
    sample = json.loads((results / "sample_manifest.json").read_text())
    if sample["sample_users"] != cfg.sample_users or sample["seed"] != cfg.seed:
        raise ValueError("Sample manifest mismatch")
    frozen = {
        "run_id": run["run_id"], "selection_split": "validation", "selection_metric": "ndcg@10",
        "validation_cutoff": cfg.validation_cutoff, "test_cutoff_reserved": cfg.test_cutoff, "test_evaluated": False,
        "parameters": {"loss_function": "YetiRank", "depth": model["selected"]["depth"],
                       "l2_leaf_reg": model["selected"]["l2_leaf_reg"], "iterations": model["selected"]["iterations"],
                       "learning_rate": cfg.learning_rate, "random_seed": cfg.seed, "thread_count": cfg.threads,
                       "max_ctr_complexity": 1},
        "checkpoint_training_threads": (model["selected"]["prefix_reuse"]["source_threads"]
                                        if model["selected"].get("prefix_reuse") else cfg.threads),
        "execution_threads_limit": cfg.threads,
        "source_commit": run["source_commit"], "archive_sha256": run["archive_sha256"],
        "sample_users": sample["sample_users"], "sample_rows": sample["sample_rows"], "user_ids_sha256": sample["user_ids_sha256"],
        "model_sha256": run["model_sha256"], "model_artifact": run["model_artifact"],
        "training_base_commit": run["git_commit"], "code_fingerprint": run["code_fingerprint"],
        "source_note": "Base HEAD at execution; code_fingerprint identifies exact sources, including any pending changes not yet included in that HEAD.",
        "data_fingerprint": run["data_fingerprint"], "config_fingerprint": run["config_fingerprint"],
        "versions": run["versions"], "python": run["python"], "config": cfg.to_dict(),
        "features": model["features"], "categorical_features": model["categorical_features"],
        "candidate_snapshots": [{"cutoff": s["cutoff"], "training": s["training"], "rows": s["rows"],
            "queries": s["queries"], "positive_candidates": s["positive_candidates"], "sha256": s["sha256"],
            "path": str((model_path.parent.parent / f"{s['cutoff']}_{'train' if s['training'] else 'validation'}.parquet").relative_to(cfg.root))}
            for s in run["snapshots"]],
        "next_stage": "Refit on approved training cutoffs plus validation cutoff with fixed iterations, then evaluate held-out test.",
    }
    write_json(results / "frozen_validation.json", frozen)
    records = [{**{k: row[k] for k in ["name", "depth", "l2_leaf_reg", "iterations", "seconds", "trained_iterations", "prefix_reuse"]}, **row["metrics"]} for row in model["grid"]]
    curves = []
    for record in records:
        checkpoint = CatBoostRanker()
        checkpoint_path = model_path.parent / f"{record['name']}.cbm"
        source_record = next(r for r in model["grid"] if r["name"] == record["name"])
        if sha256(checkpoint_path) != source_record["model_sha256"]:
            raise ValueError("Grid checkpoint checksum mismatch")
        if record["trained_iterations"] > cfg.iterations or record["iterations"] > record["trained_iterations"]:
            raise ValueError("Grid checkpoint exceeds the iteration budget")
        checkpoint.load_model(str(checkpoint_path))
        history = checkpoint.get_evals_result().get("validation", {})
        reuse = record["prefix_reuse"]
        record["source_trained_iterations"] = reuse["source_trained_iterations"] if reuse else record["trained_iterations"]
        record["execution_kind"] = "verified_prefix" if reuse else "fresh_fit"
        record["prior_source_seconds"] = reuse["original_exploratory_result"]["seconds"] if reuse else 0.0
        record["checkpoint_training_threads"] = reuse["source_threads"] if reuse else cfg.threads
        for metric, values in history.items():
            curves.extend({"configuration": record["name"], "iteration": n + 1, "internal_metric": metric, "value": value}
                          for n, value in enumerate(values[:record["trained_iterations"]]))
    pd.DataFrame(curves).to_csv(results / "internal_learning_curves.csv", index=False)
    write_json(results / "grid_provenance.json", model["grid"])
    pd.DataFrame([{k: v for k, v in r.items() if k != "prefix_reuse"} for r in records]).to_csv(results / "grid_results.csv", index=False)
    shutil.copyfile(model_path.parent / "feature_importance.csv", results / "feature_importance.csv")
    table = pd.read_csv(results / "validation_results.csv")
    lines = ["# Зафиксированный baseline: валидация", "", "Этапы 1–10. Итоговый тест не оценивался.", "",
             f"Выборка: {sample['sample_users']:,} пользователей, {sample['sample_rows']:,} взаимодействий; seed {cfg.seed}.",
             f"Валидация: история до {cfg.validation_cutoff}, горизонт {cfg.horizon_days} дней; {run['validation_users']:,} warm-active пользователей.", "",
             "Метрики усредняются по пользователям (macro); будущие цели вне кандидатов учитываются как промахи.", "",
             "| Метод | NDCG@10 | Recall@10 | MAP@10 | Recall кандидатов |", "|---|---:|---:|---:|---:|"]
    for row in table.to_dict("records"):
        lines.append(f"| {row['method']} | {row['ndcg@10']:.6f} | {row['recall@10']:.6f} | {row['map@10']:.6f} | {row['candidate_recall@200']:.6f} |")
    best = model["selected"]["metrics"]
    reference_name, reference = max(run["baselines"].items(), key=lambda kv: kv[1]["ndcg@10"])
    gain = (best["ndcg@10"] / reference["ndcg@10"] - 1) * 100
    lines.extend(["", f"Прирост NDCG@10 к лучшему контрольному методу ({reference_name}): {gain:+.2f}% относительно."])
    ordered_grid = sorted(model["grid"], key=lambda r: r["metrics"]["ndcg@10"], reverse=True)
    if len(ordered_grid) > 1:
        gap = ordered_grid[0]["metrics"]["ndcg@10"] - ordered_grid[1]["metrics"]["ndcg@10"]
        lines.append(f"Разрыв между двумя лучшими конфигурациями: {gap:.6f} NDCG@10. Это выбор по данной валидации; устойчивость различия ещё не проверена на тесте.")
    if run["ablation"]:
        delta = best["ndcg@10"] - run["ablation"]["metrics"]["ndcg@10"]
        lines.append(f"Разница с абляцией явных признаков связи: {delta:+.6f} NDCG@10. Это совместный вклад удалённой группы при неизменных кандидатах.")
    importance = pd.read_csv(results / "feature_importance.csv")
    lines.extend(["", "## Признаки выбранной модели", "", "| Признак | PredictionValuesChange |", "|---|---:|"])
    for row in importance.head(10).to_dict("records"):
        lines.append(f"| {row['feature']} | {row['importance']:.3f} |")
    lines.extend(["", "Эта важность описывает изменение предсказаний модели; она не доказывает причинный вклад и делится между коррелирующими признаками.",
                  "Определения признаков приведены в FEATURES.md. Абляция проверяет вклад явных pair-признаков и персональных баллов источников отдельным обучением."])
    lines.extend(["", "## Размеры датасетов ранжирования", "",
                  "| Срез | Назначение | Пользователей | Пар | Позитивов в кандидатах |",
                  "|---|---|---:|---:|---:|"])
    for snapshot in run["snapshots"]:
        role = "Обучение" if snapshot["training"] else "Валидация"
        lines.append(f"| {snapshot['cutoff']} | {role} | {snapshot['queries']:,} | {snapshot['rows']:,} | {snapshot['positive_candidates']:,} |")
    verification_path = results / "reload_verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        if (verification["model_sha256"] != run["model_sha256"] or verification["test_evaluated"]
                or any(abs(verification["reloaded_metrics"][k] - best[k]) > 1e-10 for k in best)):
            raise ValueError("Reload verification is stale; rerun scripts.verify_validation")
        lines.extend(["", "Повторная загрузка сохранённой модели воспроизвела все внешние метрики с допуском 1e-10; результат проверки — в reload_verification.json."])
    params = frozen["parameters"]
    lines.extend(["", f"Выбраны depth={params['depth']}, l2_leaf_reg={params['l2_leaf_reg']}, iterations={params['iterations']}, learning_rate={cfg.learning_rate}.",
        f"Признаков: {run['feature_count']}. Время текущего запуска: {run['elapsed_seconds']:.1f} с; пик RSS: {run['peak_rss_mb']:.1f} МиБ.", "",
        f"Бюджет каждой конфигурации: не более {cfg.iterations} итераций; новый запуск ограничен {cfg.threads} потоками.", "",
        "## Интерпретация протокола", "",
        "- Каждая пара KION представлена последним зарегистрированным просмотром. Это не полный журнал повторных просмотров.",
        "- Оценка условна на наличии истории и активности в будущем периоде; она не измеряет вовлечение всех пользователей.",
        "- Новые фильмы и пропуски кандидатов учитываются как промахи. Отрицательный label не означает отказ после показа.",
        "- Метаданные считаются статическими; их исторические версии отсутствуют. Пользователи без демографии не удалены.",
        "- Контрольные методы и модель используют одинаковый пул кандидатов. Ничьи разрешаются по ID фильма.",
        "- Абляция удаляет pair_*, cand_knn_*, cand_genre_* и rrf_score. Сохраняются признаки пользователя и фильма, популярность и персональный генератор кандидатов; это не полное отключение персонализации.",
        "- Early stopping CatBoost использует NDCG внутри кандидатов и значение 1 для групп без позитивных кандидатов. Внешняя метрика учитывает все цели и даёт таким группам 0; по ней выбирается конфигурация.",
        "- Выборка обучающих запросов ограничена 5000 пользователями на срез; исторические статистики используют всех выбранных пользователей.",
        "- Метрики здесь валидационные, использованные для выбора модели. Независимая итоговая оценка — отдельный этап 11.", "",
        "## Воспроизведение", "", "```bash", "python main.py download", "python main.py prepare", "python -u main.py run",
        "python -m scripts.verify_validation > results/reload_verification.json", "python -m scripts.freeze_validation", "```", "", "Параметры зафиксированы в `frozen_validation.json`; все четыре конфигурации — в `grid_results.csv`.",
        "Модель и полные датасеты находятся локально в каталоге артефактов; их контрольные суммы сохранены в манифестах."])
    lines.extend(["", "В `internal_learning_curves.csv` сохранены внутренние метрики CatBoost по итерациям.",
                  "Они нормированы внутри кандидатов (без позитивов CatBoost возвращает 1) и не заменяют внешние метрики из основной таблицы. Поле test в логе CatBoost обозначает наш валидационный eval_set, а не финальный тест.",
                  "В `grid_results.csv` iterations — размер сохранённой модели, trained_iterations — число рассмотренных итераций в пределах бюджета, source_trained_iterations — длина исходного обучения (для переиспользованного префикса может быть больше)."])
    lines.extend(["", "В grid_results.csv seconds — время текущей операции: обучение и скоринг для fresh_fit, перенос префикса и скоринг для verified_prefix.",
                  "Время старого обучения вынесено в prior_source_seconds; эти величины нельзя напрямую сравнивать как время обучения разных конфигураций."])
    for row in model["grid"]:
        reuse = row.get("prefix_reuse")
        if reuse:
            original = reuse["original_exploratory_result"]
            lines.extend(["", f"### Происхождение {row['name']}", "",
                "Префикс перенесён после проверки SHA-256 трёх обучающих датасетов и одного валидационного; исходная модель сохранена отдельно.",
                f"Исходное обучение: {reuse['source_trained_iterations']} итераций, {reuse['source_threads']} потоков, {original['seconds']:.1f} с.",
                f"Для выбора числа деревьев использованы только первые {row['trained_iterations']} точек learning curve, бюджет {reuse['prefix_budget']}; поздние точки исключены.",
                "Полный результат старой модели не участвует в сокращённой сетке. Он сохранён как исследовательский в grid_provenance.json.",
                "Затраты исходного обучения не включены во время текущего запуска. Это не суммарное время всей работы: были также прерванные попытки без полного замера. Повторный запуск с нуля использует текущий лимит потоков; возможны небольшие численные отличия."])
    (results / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(cfg.path("artifact_dir") / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    display = {"popularity_all": "Популярность: вся история", "popularity_7d": "Популярность: 7 дней",
               "itemknn": "ItemKNN", "rrf": "RRF", "catboost": "CatBoost",
               "catboost_without_personal_features": "CatBoost без явных pair-признаков"}
    ordered = table.sort_values("ndcg@10")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [display.get(x, x) for x in ordered.method]
    bars = ax.barh(labels, ordered["ndcg@10"], color=["#2463a6" if x == "catboost" else "#9aabbc" for x in ordered.method])
    ax.bar_label(bars, fmt="%.5f", padding=5)
    ax.set_xlim(0, ordered["ndcg@10"].max() * 1.25)
    ax.set_xlabel("NDCG@10 — полный целевой список")
    ax.set_title(f"KION: валидация {cfg.validation_cutoff}\n{run['validation_users']:,} пользователей с историей и целевыми просмотрами")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(results / "validation_ndcg.png", dpi=160)
    plt.close(fig)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    print(json.dumps(freeze(Config.load(args.config)), indent=2))
