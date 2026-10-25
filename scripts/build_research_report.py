"""Compile the final research narrative and figures from completed results, without fitting."""
from pathlib import Path
import json
import os

from kion_autofeatures.runtime import configure
configure(2)

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / "research_report_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

from kion.io import sha256
from kion_autofeatures.config import ResearchConfig
from kion_autofeatures.final import verify_frozen

ORDER = ("source", "source_raw", "full", "auto", "random")
LABELS = {"source": "S", "source_raw": "S + R", "full": "S + R + M",
          "auto": "S + R + A", "random": "S + R + random"}
CONTRASTS = {"auto − full (primary)": "S + R + A − S + R + M (основное)", "auto − random": "S + R + A − S + R + random",
             "auto − source": "S + R + A − S", "auto − source_raw": "S + R + A − S + R"}
SEGMENTS = {"history_1_2": "1–2 фильма в истории", "history_3_9": "3–9 фильмов в истории",
            "history_10_plus": "10 и больше фильмов", "retrieval_some_hits": "в пуле есть хотя бы одна цель",
            "retrieval_zero_hits": "в пуле нет ни одной цели"}
PATHS = {"user": "пользователь", "item": "фильм",
         "user_genres": "пользователь по жанрам кандидата", "user_directors": "пользователь по режиссёру кандидата",
         "user_countries": "пользователь по стране кандидата", "user_content_type": "пользователь по типу кандидата",
         "item_age": "фильм у зрителей того же возраста", "item_income": "фильм у зрителей того же дохода",
         "item_sex": "фильм у зрителей того же пола", "item_kids_flg": "фильм у зрителей с тем же признаком детей"}
OPERATIONS = {"count": "число просмотров", "sum_log_dur": "суммарная длительность, log1p",
              "mean_log_dur": "средняя длительность, log1p", "mean_pct": "средний % досмотра",
              "max_pct": "максимальный % досмотра", "recency": "дней с последнего просмотра",
              "share": "доля просмотров", "trend": "тренд числа просмотров"}


def describe_feature(row):
    return f"{PATHS[row.path].capitalize()}: {OPERATIONS[row.operation]}"


def describe_window(row):
    if row.operation == "trend":
        return f"{row.window} / {int(row.long_window)} дн."
    return "вся история" if str(row.window) == "all" else f"{row.window} дн."


def read(path):
    return json.loads(Path(path).read_text())


def load_evidence():
    research = ResearchConfig.load(ROOT / "autofeatures.toml")
    required = ("inputs", "search", "selection", "validation", "frozen", "test",
                "test_intervals", "test_verification", "validation_verification", "random_subsets")
    for name in required:
        if not (research.results / f"{name}.json").is_file():
            raise FileNotFoundError(f"Complete the experiment first: {name}.json")
    frozen = verify_frozen(research)
    test = read(research.results / "test.json")
    if not test["test_evaluated"] or test["identity"]["frozen_seal"] != frozen["seal"]:
        raise ValueError("Test results do not belong to the frozen method")
    return research, {name: read(research.results / f"{name}.json") for name in required}


def markdown_table(frame, columns, labels=None):
    labels = labels or columns
    rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{value:.5f}" if isinstance(value, float) else str(value) for value in values) + " |")
    return "\n".join(rows)


def build():
    research, evidence = load_evidence()
    directory = research.results
    output = directory / "report"
    output.mkdir(exist_ok=True)
    validation = pd.read_csv(directory / "validation_results.csv").set_index("variant")
    test = pd.read_csv(directory / "test_results.csv").set_index("variant")
    selection = evidence["selection"]
    baseline = read(ROOT / "results" / "report_summary.json")
    intervals = evidence["test_intervals"]
    primary = intervals["primary_auto_minus_full"]["ndcg@10"]
    paired = {"auto − full (primary)": primary, **{
        "auto − " + name: value["ndcg@10"] for name, value in intervals["diagnostic_auto_minus"].items()}}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    values = [test.loc[name, "ndcg@10"] for name in ORDER]
    axes[0].barh([LABELS[n] for n in ORDER], values, color=["#8a9aa8", "#8a9aa8", "#325878", "#d77d28", "#54876a"])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, .18), xlabel="Test macro NDCG@10", title="Same users and 200-candidate pool")
    for index, value in enumerate(values):
        axes[0].text(value + .002, index, f"{value:.5f}", va="center", fontsize=10)
    for index, (name, value) in enumerate(paired.items()):
        axes[1].errorbar(value["mean"], index,
                        xerr=[[value["mean"] - value["lower"]], [value["upper"] - value["mean"]]],
                        fmt="o", capsize=4, color="#d77d28" if index == 0 else "#657b8a")
    axes[1].set_yticks(range(len(paired)), paired)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#333333", linewidth=1, linestyle="--")
    axes[1].xaxis.set_major_locator(MultipleLocator(.002))
    axes[1].set(xlabel="Paired ΔNDCG@10; 95% user-bootstrap CI", title="Secondary comparisons: unadjusted")
    fig.savefig(output / "test_comparison.png", dpi=160)
    plt.close(fig)
    prefixes = pd.DataFrame([{"size": p["size"], "ndcg": p["metrics"]["ndcg@10"]} for p in selection["prefixes"]])
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(prefixes["size"], prefixes.ndcg, "o-", color="#325878")
    ax.scatter([16], [prefixes.iloc[0].ndcg], color="#d77d28", s=80, zorder=3, label="Chosen: within 0.0001 of best")
    ax.set(xticks=prefixes["size"], xlabel="Automatic features (plus S + R)", ylabel="Inner macro NDCG@10",
           title="Selection on 1,000 users, 2 August only")
    ax.legend(fontsize=9)
    fig.savefig(output / "inner_selection.png", dpi=160)
    plt.close(fig)
    subsets = evidence["random_subsets"]
    if subsets["test_evaluated"] or not all(subsets["comparability"].values()):
        raise ValueError("Random-subset diagnostic must reuse the inner split and never the test")
    runs = pd.DataFrame(subsets["results"])
    draws = runs.loc[runs.kind == "random_draw"].sort_values("ndcg@10").reset_index(drop=True)
    seeds = runs.loc[runs.kind.isin(["selected", "selected_seed"])]
    chosen_ndcg = runs.loc[runs.kind == "selected", "ndcg@10"].iloc[0]
    frozen_ndcg = runs.loc[runs.kind == "frozen_random", "ndcg@10"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    ax.axvline(chosen_ndcg, color="#9aa3ab", linewidth=1, linestyle="--", zorder=1)
    # Alternate small vertical offsets so close values stay visible as separate dots.
    ax.scatter(draws["ndcg@10"], 2 + (draws.index % 3 - 1) * .14, s=56, color="#bcc3ca",
               edgecolor="#6b7883", linewidth=.8, zorder=2)
    ax.scatter([frozen_ndcg], [1], marker="D", s=72, color="#54876a", edgecolor="white", linewidth=1.5, zorder=3)
    other = seeds.loc[seeds.kind == "selected_seed", "ndcg@10"].sort_values().to_numpy()
    ax.scatter(other, [(-1) ** n * .14 for n in range(len(other))], s=48, facecolor="white",
               edgecolor="#d77d28", linewidth=1.6, zorder=2, label="Other CatBoost seeds")
    ax.scatter([chosen_ndcg], [0], s=80, color="#d77d28", edgecolor="white", linewidth=1.5, zorder=3,
               label="Seed 42 (frozen)")
    ax.annotate(f"{chosen_ndcg:.5f}", (chosen_ndcg, 0), xytext=(-10, -24), textcoords="offset points",
                ha="right", fontsize=9, color="#333333")
    ax.annotate(f"{frozen_ndcg:.5f}", (frozen_ndcg, 1), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color="#333333")
    ax.set_yticks([0, 1, 2], ["Selected A", "Frozen random\n(test control)", f"{len(draws)} new\nrandom sets"])
    ax.set_ylim(-.5, 2.5)
    ax.grid(axis="x", color="#e6e6e6", linewidth=.8)
    ax.set_axisbelow(True)
    ax.set(xlabel="Inner macro NDCG@10 (2 August, 1,000 users)",
           title=f"Selected {subsets['protocol']['size']} features vs random sets of the same size")
    ax.legend(title="Selected A", title_fontsize=8, fontsize=8, loc="lower left",
              bbox_to_anchor=(1.01, 0), frameon=False)
    fig.savefig(output / "random_subsets.png", dpi=160)
    plt.close(fig)
    subset_rows = []
    for metric in ("ndcg@10", "recall@10", "map@10"):
        value = subsets["summary"][metric]
        subset_rows.append({
            "metric": {"ndcg@10": "NDCG@10", "recall@10": "Recall@10", "map@10": "MAP@10"}[metric],
            "selected": value["selected"],
            "random": f"{value['random_mean']:.5f} ± {value['random_std']:.5f}",
            "range": f"[{value['random_min']:.5f}; {value['random_max']:.5f}]",
            "rank": f"{value['selected_among_draws']['rank']} из {value['selected_among_draws']['of']}",
            "frozen": f"{value['frozen_random']:.5f} ({value['frozen_random_among_draws']['rank']} из {value['frozen_random_among_draws']['of']})",
            "seeds": f"[{value['selected_seed_min']:.5f}; {value['selected_seed_max']:.5f}]"})
    subset_table = markdown_table(pd.DataFrame(subset_rows), ["metric", "selected", "random", "range", "rank", "frozen", "seeds"],
                                  ["Метрика", "Отобранный A", "Случайные: среднее ± sd", "Случайные: диапазон",
                                   "Место A", "Контрольный random (место)", "A на 5 seed: диапазон"])
    subset_ndcg = subsets["summary"]["ndcg@10"]
    subset_mean = subsets["paired_selected_minus"]["mean_of_draws"]["ndcg@10"]
    subset_excluded = subsets["paired_ndcg_intervals_excluding_zero"]
    subset_rank = subset_ndcg["selected_among_draws"]
    subset_frozen_rank = subset_ndcg["frozen_random_among_draws"]
    subset_ratio = subset_ndcg["random_std"] / subset_ndcg["selected_seed_std"]
    subset_overlap = runs.loc[runs.kind == "random_draw", "overlap_with_selected"]
    quality = []
    for name in ORDER:
        quality.append({"variant": LABELS[name], "features": len(evidence["test"]["models"][name]["identity"]["columns"]),
                        "validation": validation.loc[name, "ndcg@10"], **{
                            metric: test.loc[name, metric] for metric in ("ndcg@10", "recall@10", "map@10")}})
    quality_table = markdown_table(pd.DataFrame(quality), ["variant", "features", "validation", "ndcg@10", "recall@10", "map@10"],
                                  ["Вариант", "Колонок", "NDCG@10, валидация", "NDCG@10, тест", "Recall@10, тест", "MAP@10, тест"])
    costs = []
    for name in ORDER:
        record = evidence["test"]["models"][name]
        costs.append({"variant": LABELS[name],
                      "fit": f"{record['fit_seconds']:.0f}" if record.get("fit_seconds") is not None else "готовая модель",
                      "pool": f"{record['training_pool_seconds']:.1f}" if "training_pool_seconds" in record else "нет",
                      "predict": f"{record['predict_seconds_median']:.2f}", "size": f"{record['model_bytes'] / 1024**2:.2f}"})
    cost_table = markdown_table(pd.DataFrame(costs), ["variant", "pool", "fit", "predict", "size"],
                               ["Вариант", "Подготовка Pool, с", "Обучение, с", "Предсказание, с", "Модель, МиБ"])
    resources = []
    for stage, label in (("search", "Поиск признаков"), ("validate", "Валидация вариантов"), ("test", "Финальное обучение и тест")):
        audit = evidence["test_verification"] if stage == "test" else evidence["validation_verification"]
        value = audit["resources"][stage]
        resources.append({"stage": label, "minutes": f"{value['seconds'] / 60:.1f}",
                          "rss": f"{value['peak_tree_rss_mb']:.0f}", "over_limit": value["memory_limit_exceeded"]})
    stage_table = markdown_table(pd.DataFrame(resources), ["stage", "minutes", "rss"], ["Этап", "Минуты", "Пик памяти, МиБ"])
    memory_note = ("ни один этап не превысил лимит" if not any(r["over_limit"] for r in resources)
                   else "часть этапов превысила лимит")
    stage_minutes = sum(float(r["minutes"]) for r in resources)
    selected = pd.read_csv(directory / "selected_features.csv")
    selected = selected.loc[selected["set"] == "selected"].reset_index(drop=True)
    selected["description"] = [describe_feature(row) for row in selected.itertuples()]
    selected["window_text"] = [describe_window(row) for row in selected.itertuples()]
    selected["code"] = [f"`{name}`" for name in selected.feature]
    selected_table = markdown_table(selected, ["description", "window_text", "code"], ["Что считает", "Окно", "Имя в коде"])
    contrast_table = markdown_table(pd.DataFrame([{"contrast": CONTRASTS.get(name, name), **value} for name, value in paired.items()]),
                                    ["contrast", "mean", "lower", "upper"], ["Сравнение", "ΔNDCG@10", "95% CI: от", "95% CI: до"])
    prefix_table = markdown_table(prefixes, ["size", "ndcg"], ["Число автопризнаков", "NDCG@10 на 2 августа"])
    segments = pd.read_csv(directory / "test_segments.csv")
    pivot = segments.pivot(index="segment", columns="variant", values="ndcg@10").reindex(list(SEGMENTS))[list(ORDER)]
    segment_users = segments.drop_duplicates("segment").set_index("segment").users
    segment_frame = pd.DataFrame({"segment": [SEGMENTS[s] for s in pivot.index],
                                  "users": [int(segment_users[s]) for s in pivot.index],
                                  **{name: pivot[name].to_numpy() for name in ORDER}})
    segment_table = markdown_table(segment_frame, list(segment_frame.columns),
                                   ["Сегмент", "Пользователей", *[LABELS[name] for name in ORDER]])
    reference_gap = baseline["paired_ndcg_difference"]
    zero_share = baseline["zero_positive_candidates_users"] / baseline["test_users"]
    over_raw = intervals["diagnostic_auto_minus"]["source_raw"]["ndcg@10"]
    over_random = intervals["diagnostic_auto_minus"]["random"]["ndcg@10"]
    raw_gain = test.loc["source_raw", "ndcg@10"] - test.loc["source", "ndcg@10"]
    manual_gain = test.loc["full", "ndcg@10"] - test.loc["source_raw", "ndcg@10"]
    grid = pd.read_csv(ROOT / "results" / "grid_results.csv")
    budget_hits = int((grid.trained_iterations - grid.iterations <= 2).sum())
    params = evidence["frozen"]["parameters"]
    curves = pd.read_csv(ROOT / "results" / "internal_learning_curves.csv")
    curve = curves.loc[(curves.configuration == f"depth{params['depth']}_l2{params['l2_leaf_reg']}")
                       & curves.internal_metric.str.startswith("NDCG")].set_index("iteration").value
    late_gain = curve.iloc[-1] - curve.loc[curve.index.max() - 50]
    test_users = f"{baseline['test_users']:,}".replace(",", " ")
    zero_users = f"{baseline['zero_positive_candidates_users']:,}".replace(",", " ")
    selected_better, random_better = subset_excluded["selected_better"], subset_excluded["random_better"]
    if random_better:
        excluded_text = (f"Парный интервал исключает ноль в {selected_better + random_better} сравнениях из "
                         f"{subsets['protocol']['draws']}: {selected_better} в пользу отобранного набора, "
                         f"{random_better} в пользу случайного.")
    else:
        excluded_text = (f"Парный интервал исключает ноль в {selected_better} сравнениях из "
                         f"{subsets['protocol']['draws']}, и все они в пользу отобранного набора "
                         "против самых слабых случайных.")
    text = f"""# Автоматическое построение признаков для рекомендаций KION

Вопрос задания: можно ли заменить ручные признаки градиентного бустинга
автоматически построенными без потери качества рекомендаций. Здесь проверен
один способ: признаки-агрегаты по истории просмотров генерируются перебором,
а затем отбираются по важности.

| Критерий задания | Где в отчёте |
|---|---|
| Сильный baseline без утечек | [разделы 1–3](#1-задача-и-данные); подробности в [приложении](docs/baseline_details.md) |
| Исследовательская идея: автоматические признаки | [раздел 4](#4-автоматические-признаки-генерация-и-отбор) |
| Сравнение на одном разбиении, затраты и масштабируемость | [раздел 5](#5-результаты) и [раздел 6](#6-затраты-и-масштабируемость) |
| Анализ: что сработало, что нет и почему | [раздел 5](#5-результаты) и [раздел 7](#7-выводы-и-ограничения) |

## Кратко

- **Baseline.** CatBoostRanker на 82 признаках ранжирует до 200 кандидатов.
  Тестовая NDCG@10 равна **{baseline['test_ndcg']:.5f}**, это на {baseline['relative_gain_percent']:.1f}% выше
  популярности за 7 дней (разница {reference_gap['mean']:+.4f}, 95% CI
  [{reference_gap['lower']:+.4f}; {reference_gap['upper']:+.4f}]).
- **Автоматические признаки.** 16 отобранных агрегатов вместо 54 ручных
  (44 входные колонки вместо 82) дают **{test.loc['auto', 'ndcg@10']:.5f}** против
  {test.loc['full', 'ndcg@10']:.5f} у baseline. Разница {primary['mean']:+.5f}, 95% CI
  [{primary['lower']:+.5f}; {primary['upper']:+.5f}]: качество близкое, но ни превосходство,
  ни равенство не доказаны.
- **Отбор не лучше случайного.** Контрольный набор из 16 случайных агрегатов
  получил на тесте {test.loc['random', 'ndcg@10']:.5f}, больше отобранного. На прошлых данных,
  без теста, отобранный набор занимает {subset_rank['rank']}-е место из {subset_rank['of']} среди
  случайных наборов того же размера.
- **Главное ограничение в кандидатах.** У {zero_share:.0%} пользователей в пуле нет ни
  одной будущей цели; идеальное ранжирование пула дало бы NDCG@10
  {baseline['oracle_candidate_ndcg']:.2f}.

Итог: автоматически построенные агрегаты могут заменить ручные почти без
потери качества и с меньшим числом входов, но предложенный отбор находит
наборы не лучше случайных.

## 1. Задача и данные

Данные: [KION](https://github.com/irsafilo/KION_DATASET), ревизия `f69775b`.
Из 962 179 пользователей случайно выбраны 100 000 (seed 42) со всей их историей:
572 621 запись и 10 082 фильма. Полный релиз содержит 5,48 млн записей с 13 марта
по 22 августа 2021 года. Хеши архива и выборки лежат в
[sample_manifest.json](results/sample_manifest.json).

Задача поставлена как ранжирование: выбрать для пользователя top-10 из ещё не
просмотренных фильмов. Позитив означает запись о просмотре пары
пользователь–фильм в следующие 7 дней, независимо от процента досмотра. Лога
показов в KION нет, поэтому метка 0 означает «не смотрел», а не «увидел и
отказался». Отсюда YetiRank и метрики ранжирования (NDCG@10, Recall@10, MAP@10)
вместо LogLoss и ROC-AUC.

Особенность KION: каждая строка описывает пару пользователь–фильм с датой
**последнего** просмотра. Если последний просмотр пары попал в будущее, ранние
просмотры этой пары не видны. Метаданные пользователей и фильмов статичны и
считаются известными на дату прогноза.

Качество оценивается на активных пользователях: у них есть история до даты
среза и хотя бы один просмотр в следующую неделю. На тесте таких
{test_users}; 5 870 пользователей без истории в оценку не входят. 873 цели
приходятся на фильмы, которых не было в истории; они остаются в знаменателях
метрик как промахи.

## 2. Разбиение по времени и защита от утечек

| Срез (история строго до) | Неделя меток | Роль |
|---|---|---|
| 19 июля 2021 | 19–25 июля | Обучение; обучение при отборе признаков |
| 26 июля | с 26 июля по 1 августа | Обучение; обучение при отборе признаков |
| 2 августа | 2–8 августа | Обучение; оценка при отборе признаков |
| 9 августа | 9–15 августа | Валидация: выбор параметров и сравнение вариантов |
| 16 августа | 16–22 августа | Итоговый тест, открыт один раз |

- На каждом срезе признаки, кандидаты и ItemKNN считаются только по записям
  с датой строго раньше среза.
- Метки следующей недели присоединяются после построения кандидатов и
  признаков; функции признаков метки не принимают.
- Параметры baseline, отобранные признаки и правила сравнения зафиксированы
  до открытия теста, после теста ничего не менялось.
- Автотесты проверяют, что изменение будущих событий не меняет признаки.

На обучение берётся до 5 000 случайных активных пользователей на срез, а
агрегаты считаются по истории всех 100 000. Запросы без позитива среди
кандидатов исключаются только из обучения. Обучающая выборка для валидации
содержит 2,22 млн пар и 11 120 запросов; финальное обучение добавляет срез
9 августа, итого 2,94 млн пар и 14 704 запроса.

## 3. Baseline: кандидаты и ручные признаки

**Кандидаты.** Четыре источника: популярность за всю историю, популярность
за 7 дней, ItemKNN и жанровый профиль пользователя. Каждый предлагает до 100
фильмов. Первые 10 от каждого источника сохраняются, остальные места до 200
заполняются по RRF (k = 60). Уже просмотренные фильмы исключаются. ItemKNN
использует косинусную близость по последним 100 фильмам пользователя со
сжатием co/(co + 10) и до 100 соседей на фильм. Все варианты моделей
ранжируют один и тот же пул.

**Модель.** CatBoostRanker (YetiRank), depth={params['depth']}, l2={params['l2_leaf_reg']},
learning_rate={params['learning_rate']}, {params['iterations']} деревьев. Параметры выбраны на валидации
по сетке depth ∈ {{6, 8}}, l2 ∈ {{3, 10}} с бюджетом 300 деревьев. В {budget_hits} из {len(grid)}
конфигураций лучшая итерация пришлась на конец бюджета. У выбранной
конфигурации внутренняя NDCG за последние 50 деревьев выросла лишь на
{late_gain:.4f}, но больший бюджет мог бы немного улучшить baseline. В одной
конфигурации сетки взяты первые 300 деревьев более раннего запуска; выбранная
и все финальные модели обучены заново
([grid_provenance.json](results/grid_provenance.json)).

**Признаки.** 82 колонки в трёх группах:

| Группа | Колонок | Что входит |
|---|---:|---|
| S: баллы источников | 13 | баллы, ранги и флаги присутствия четырёх источников кандидатов, RRF |
| R: статические поля | 15 | демография пользователя; тип, страна, режиссёр, жанры, ID, год и возрастной рейтинг фильма |
| M: ручные агрегаты | 54 | счётчики, средняя длительность и % досмотра, давность и тренды пользователя и фильма за 7/30/90 дней и всю историю; связи пользователя с жанром, режиссёром и типом кандидата |

Определения приведены в [FEATURES.md](results/FEATURES.md). Прямых признаков
пары («сколько раз пользователь смотрел этот фильм») нет: просмотренные фильмы
исключены из кандидатов, а повторные просмотры в KION не видны. Сравнение
с популярностью и ItemKNN, абляции групп и важности признаков приведены в
[приложении](docs/baseline_details.md).

## 4. Автоматические признаки: генерация и отбор

**Идея.** Почти каждый ручной признак из M устроен одинаково: «что считать»
по «чьей истории» за «какой период». Например, число просмотров фильмов того же
жанра пользователем за 30 дней. Вместо того чтобы выбирать эти три части
вручную, программа перебирает все их сочетания, а полезные отбирает модель.
Подход близок к
[Deep Feature Synthesis](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf)
(Kanter, Veeramachaneni, 2015); идея ограничивать пространство и стоимость
поиска взята из [OpenFE](https://proceedings.mlr.press/v202/zhang23ay.html)
(Zhang et al., 2023). Сами алгоритмы этих статей не реализованы.

**Перебор даёт 504 признака**; при их построении метки не используются:

- 10 связей: история пользователя; история фильма; история пользователя по
  жанру, режиссёру, стране и типу кандидата; история фильма среди зрителей
  того же возраста, дохода, пола и признака детей.
- 8 окон: 3, 7, 10, 14, 30, 60, 90 дней и вся история.
- Операции: число просмотров, сумма и среднее log(1 + длительность), средний
  и максимальный % досмотра (400 признаков); давность последнего просмотра (10);
  доля относительно всей истории пользователя или фильма (64); тренд
  (count(короткое окно) + 1) / (count(длинное окно) + 1) для пар 3/14, 7/30
  и 10/60 дней (30).

Для жанров одна запись делится между жанрами фильма с весом 1/число жанров;
для режиссёра и страны берётся первый указанный. По смыслу перебор покрывает
43 из 54 ручных формул; не покрыты флаги пропусков, число активных дней,
разнообразие жанров и срок с первого просмотра.

**Отбор** использует только прошлые срезы, без валидации и теста:

1. 1 000 пользователей на срез: обучение на 19 и 26 июля (400 тыс. пар),
   оценка на 2 августа (200 тыс. пар).
2. Удаляются константы и точные копии колонок S, остаётся 502 признака.
3. Короткая модель (120 деревьев, depth=6) на S + R и всех 502 признаках
   ранжирует их по PredictionValuesChange на обучающих данных.
4. Первые 16, 32 и 64 признака по важности обучаются полной моделью и
   оцениваются на 2 августа. Берётся самый короткий набор в пределах 0.0001
   от лучшего.

{prefix_table}

Выбран размер 16. Контрольный случайный набор того же размера выбран заранее
(seed 42) равномерно из тех же 502 признаков. Затем все варианты сравниваются
на валидации 9 августа, метод фиксируется, и только после этого открывается
тест. Правила записаны в [протоколе](docs/automatic_features_protocol.md) до запуска.

![Выбор размера при отборе](results/autofeatures/main/report/inner_selection.png)

## 5. Результаты

Варианты различаются только набором признаков; кандидаты, пользователи, метки
и параметры CatBoost одинаковы.

| Обозначение | Колонок | Состав |
|---|---:|---|
| S | 13 | только баллы источников |
| S + R | 28 | плюс статические поля |
| S + R + M | 82 | плюс 54 ручных агрегата (baseline) |
| S + R + A | 44 | плюс 16 отобранных автоматических агрегатов |
| S + R + random | 44 | плюс 16 случайных из тех же 504 автоматических признаков |

{quality_table}

Метрики усреднены по пользователям. Recall@10 делится на полное число будущих
целей, MAP@10 на min(10, число целей); идеальная DCG тоже строится по всем
целям, включая не попавшие в пул.

Парные разницы на тесте (bootstrap по пользователям, 1 000 повторов):

{contrast_table}

![Качество и интервалы](results/autofeatures/main/report/test_comparison.png)

- **A вместо M.** Замена 54 ручных агрегатов на 16 автоматических почти не
  меняет NDCG: интервал разницы включает ноль. Граница неухудшения заранее не
  задавалась, поэтому эквивалентность не утверждается. На валидации картина
  та же: −0.00022, 95% CI [−0.00164; +0.00118].
- **Вклад агрегатов поверх S + R не подтверждён.** S + R + A выше S + R на
  {over_raw['mean']:+.5f}, но интервал [{over_raw['lower']:+.5f}; {over_raw['upper']:+.5f}] включает ноль. Большую
  часть выигрыша полной модели над S даёт уже R: S → S + R {raw_gain:+.5f},
  S + R → S + R + M {manual_gain:+.5f}. Без S (на валидации) ручные и
  автоматические агрегаты тоже близки: R + M 0.14108, R + A 0.14087.
- **Случайный набор выиграл у отобранного** ({over_random['mean']:+.5f}, интервал ниже нуля).
  Это вторичное сравнение без поправки на множественность; основная гипотеза
  после теста не менялась.
- Интервалы учитывают только разброс по пользователям при фиксированных
  моделях и одной неделе, без разброса от переобучения и смены периода.

**Пул кандидатов.** На тесте 2,28 млн пар, recall кандидатов@200 равен 0.544.
У {zero_users} пользователей ({zero_share:.1%}) в пул не попала ни одна цель, и для
них метрики любого ранжировщика равны нулю. Идеальное ранжирование этого пула
дало бы NDCG@10 = {baseline['oracle_candidate_ndcg']:.3f}, так что улучшать можно и отбор
кандидатов, и порядок.

### Какие признаки выбраны

{selected_table}

Задание спрашивает, почему окно именно семь дней. Отбор выбрал окна, которых
нет среди ручных (7/30/90): 6 из 16 агрегатов считаются за 3 дня, ещё 4
являются трендами 10/60 дней. Половина набора описывает суммарную длительность
просмотров. Чаще всего выбраны статистики фильма, в целом и среди зрителей той
же демографии, затем жанровые предпочтения пользователя. С ручной формулой
напрямую совпадает только доля жанров за 30 дней (`pair_genre_share_30`, на
10 000 реальных пар расхождение не больше 6e-8). Режиссёров в наборе нет, но
это результат одного отбора, а не довод против таких признаков.

На A приходится 48% важности валидационной модели, на S 47%, на R 4%.
Важность делится между коррелированными колонками и не равна вкладу в качество.

### Отбор против случайных наборов того же размера

Проигрыш одному случайному набору мог оказаться случайностью. Поэтому после
теста, не трогая тестовые данные, на тех же данных, что и при отборе (обучение
19/26 июля, оценка 2 августа, 1 000 пользователей), обучены ещё {subsets['protocol']['draws']} случайных
наборов по {subsets['protocol']['size']} агрегатов. Они выбраны по тому же правилу, что и контрольный,
но с другими seed. Отобранный набор дополнительно обучен с {subsets['protocol']['extra_seeds_for_selected']} другими seed
CatBoost, чтобы отделить влияние признаков от случайности обучения. Повторное
обучение отобранного набора совпало с результатом отбора до 1e-10, так что
сравнение идёт на тех же данных.

{subset_table}

![Отобранный набор среди случайных](results/autofeatures/main/report/random_subsets.png)

- Отобранный набор занимает **{subset_rank['rank']}-е место из {subset_rank['of']}** по NDCG@10:
  {subset_rank['draws_at_least']} из {subsets['protocol']['draws']} случайных не хуже, эмпирическое p = {subset_rank['empirical_p_one_sided']:.2f}. Средний
  случайный набор хуже на {subset_mean['mean']:.5f}, но интервал
  [{subset_mean['lower']:+.5f}; {subset_mean['upper']:+.5f}] включает ноль. {excluded_text}
- Разброс между случайными наборами (sd {subset_ndcg['random_std']:.5f}) примерно в {subset_ratio:.0f} раз
  больше, чем разброс отобранного набора по seed (sd {subset_ndcg['selected_seed_std']:.5f}). Выбор
  агрегатов влияет на качество, но отбор по важности использует это слабо:
  он отсекает худшие наборы, но не находит лучшие.
- Случайные наборы почти не пересекаются с отобранным ({subset_overlap.min()}–{subset_overlap.max()} общих
  признака из {subsets['protocol']['size']}) и при этом дают близкое качество: многие агрегаты
  взаимозаменяемы.
- Контрольный случайный набор и здесь выше отобранного ({subset_ndcg['frozen_random']:.5f},
  {subset_frozen_rank['rank']}-е место из {subset_frozen_rank['of']}): на тест попал удачный, а не типичный
  случайный набор.

Ограничения проверки: 1 000 пользователей и одна неделя дают парные интервалы
около ±0.005; размер 16 выбирался на этих же данных, что скорее завышает
отобранный набор. Проверка сделана после теста и объясняет результат, но не
заменяет новый отложенный период. Результаты:
[random_subsets.json](results/autofeatures/main/random_subsets.json) и
[CSV](results/autofeatures/main/random_subsets.csv).

### NDCG@10 по сегментам пользователей на тесте

{segment_table}

Различия между вариантами внутри сегментов малы, а интервалы по сегментам не
считались, поэтому победителя по сегментам не объявляем. У всех вариантов
качество падает с длиной истории пользователя.

## 6. Затраты и масштабируемость

Замеры на Apple M4 (16 ГБ), Python 3.14.7, CatBoost 1.2.10. Этапы шли
последовательно, в 2 потока и с лимитом памяти 6 ГБ; {memory_note}.

{stage_table}

Поиск, валидация и финальный этап для автоматических признаков заняли около
{stage_minutes:.0f} минут без baseline и вспомогательных проверок. Проверка
случайными наборами заняла ещё {subsets['resources']['seconds'] / 60:.0f} минут.

{cost_table}

- Генерация всех 504 признаков занимает 17–19 с на матрицу из 200 тыс. пар.
  32 колонки, нужные для валидации (A и random), строятся за 5,4 с на
  обучающий срез и 14,6 с на 2,17 млн оценочных пар. Это время добавляется
  к построению кандидатов и признаков S и R.
- Признаков на 46% меньше, но модель того же размера и предсказывает почти так
  же быстро: число и глубина деревьев не меняются. Меньшее число колонок не
  делает дешевле весь конвейер.
- Время предсказания измерено как медиана трёх прогонов на готовой тестовой
  матрице из 2,28 млн пар, без построения кандидатов и признаков. Полная модель
  переиспользована, её исходное обучение заняло 1049 с вместе с Pool.

Признаки считаются группировками по истории и проецируются на пары батчами по
20 тыс., полная матрица пользователь × фильм не строится. Стоимость растёт
с числом событий, жанров на фильм, окон и пар-кандидатов. Для миллиардов
событий нужны инкрементальные агрегаты или хранилище признаков; это, как и
задержка онлайн-сервиса, здесь не проверялось.

## 7. Выводы и ограничения

1. Baseline на ручных признаках построен без утечек и на {baseline['relative_gain_percent']:.0f}% лучше
   популярности за 7 дней.
2. Автоматически сгенерированные агрегаты могут заменить ручные: 16 отобранных
   дают близкое качество при 44 входах вместо 82.
3. Предложенный отбор по важности не лучше случайного выбора из тех же
   признаков: {subset_rank['rank']}-е место из {subset_rank['of']} на прошлых данных и проигрыш контрольному
   набору на тесте. Пользу приносит сам перебор, а не отбор.
4. Главное ограничение системы в кандидатах: у {zero_share:.0%} пользователей
   в пуле нет ни одной цели, и это ограничивает любой ранжировщик.

Что проверить дальше, с новым протоколом и новым отложенным периодом: отбор
с учётом корреляции признаков (например, жадное добавление по приросту качества,
как в OpenFE), отбор на нескольких прошлых срезах, распределение случайных
контролей на валидации, улучшение кандидатов. Текущий тест для подбора больше
использовать нельзя.

**Ограничения.** Один датасет, одна выборка пользователей, один seed отбора и
одна тестовая неделя. Хранится только последний просмотр пары, нет показов
и явных негативов, метаданные статичны, оценка только на активных
пользователях. Тестовый результат baseline был известен до построения
автоматических признаков; правила их сравнения зафиксированы до тестовой
оценки, но это слабее полностью скрытого отложенного набора. Офлайн-метрики
не гарантируют прироста CTR или удержания.

**Проверки.** Все 12 моделей валидации и теста повторно загружаются и дают те же
прогнозы и метрики (atol 1e-10); порядок пар, схемы признаков и контрольные
суммы артефактов сверяются, файлы baseline не менялись. Выполненный
[ноутбук](notebooks/kion_research.ipynb) не обучает моделей: он проверяет все
сохранённые модели и пересчитывает агрегаты на реальной истории, в том числе
при искажённых будущих событиях. Автотесты на синтетических данных проверяют
граничные случаи и утечки.

## 8. Воспроизведение

Окружение: Python 3.14.7, CPU, точные версии в `requirements.lock`. Данные,
модели и кэши (около 1,5 ГБ) в Git не хранятся, поэтому на новой машине нужен
полный прогон, около 5 часов CPU: baseline около 2 ч, автоматические признаки
около 2 ч, проверка случайными наборами около 1 ч.

```bash
python3.14 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q

python main.py download
python main.py prepare
python -u main.py run
python -m scripts.verify_validation
python -m scripts.freeze_validation
python main.py lock-test
python -u main.py test
python -m scripts.feature_ablation

python -m kion_autofeatures reproduce --name reproduction_01
python -m scripts.random_subsets --name reproduction_01
```

`reproduce` повторяет все шаги автоматических признаков под новым именем и не
трогает опубликованные результаты. Повторная оценка на том же тесте проверяет
воспроизводимость, но нового независимого эксперимента не даёт. Если
артефакты основного запуска уже есть локально, их можно проверить без обучения:

```bash
python -m kion_autofeatures verify
python -m scripts.build_research_report
python -m scripts.build_research_notebook --execute
```

Пошаговые команды и продолжение после прерывания описаны в
[инструкции](docs/automatic_features_usage.md), подробности baseline в
[приложении](docs/baseline_details.md). Промежуточные записи эксперимента:
[валидация](results/autofeatures/main/VALIDATION.md),
[тест](results/autofeatures/main/TEST.md).
"""
    (ROOT / "REPORT.md").write_text(text)
    source_files = [directory / f"{name}.json" for name in evidence]
    source_files += [directory / name for name in ("validation_results.csv", "test_results.csv", "test_segments.csv",
                                                   "selected_features.csv", "random_subsets.csv")]
    source_files += [ROOT / "results" / "report_summary.json", Path(__file__)]
    summary = {"frozen_seal": evidence["frozen"]["seal"], "source_files": {
        str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "quality": quality, "primary_auto_minus_full": primary,
        "selected_count": len(selection["selected"]), "fit_called": False,
        "report_sha256": sha256(ROOT / "REPORT.md"),
        "figures": {p.name: sha256(p) for p in sorted(output.glob("*.png"))}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return ROOT / "REPORT.md"


if __name__ == "__main__":
    print(build())
