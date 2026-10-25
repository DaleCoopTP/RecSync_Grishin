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
        rows.append("| " + " | ".join(f"{value:.8f}" if isinstance(value, float) else str(value) for value in values) + " |")
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
            "metric": metric, "selected": value["selected"],
            "random": f"{value['random_mean']:.5f} ± {value['random_std']:.5f}",
            "range": f"{value['random_min']:.5f} – {value['random_max']:.5f}",
            "rank": f"{value['selected_among_draws']['rank']} из {value['selected_among_draws']['of']}",
            "frozen": f"{value['frozen_random']:.5f} ({value['frozen_random_among_draws']['rank']} из {value['frozen_random_among_draws']['of']})",
            "seeds": f"{value['selected_seed_min']:.5f} – {value['selected_seed_max']:.5f}"})
    subset_table = markdown_table(pd.DataFrame(subset_rows), ["metric", "selected", "random", "range", "rank", "frozen", "seeds"],
                                  ["Метрика", "Выбранный A", "Случайные: среднее ± sd", "Случайные: min – max",
                                   "Место A", "Frozen random (место)", "A на 5 seed: min – max"])
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
                                  ["Вариант", "Колонок", "Val NDCG@10", "Test NDCG@10", "Test Recall@10", "Test MAP@10"])
    costs = []
    for name in ORDER:
        record = evidence["test"]["models"][name]
        costs.append({"variant": LABELS[name], "fit": f"{record['fit_seconds']:.2f}" if record.get("fit_seconds") is not None else "готовая модель",
                      "pool": f"{record['training_pool_seconds']:.3f}" if "training_pool_seconds" in record else "—",
                      "predict": f"{record['predict_seconds_median']:.3f}", "bytes": record["model_bytes"]})
    cost_table = markdown_table(pd.DataFrame(costs), ["variant", "pool", "fit", "predict", "bytes"],
                               ["Вариант", "Train Pool, с", "Fit, с", "Predict, с", "Модель, байт"])
    resources = []
    for stage in ("search", "validate", "test"):
        audit = evidence["test_verification"] if stage == "test" else evidence["validation_verification"]
        value = audit["resources"][stage]
        resources.append({"stage": stage, "minutes": f"{value['seconds']/60:.2f}",
                          "rss": f"{value['peak_tree_rss_mb']:.1f}", "over_limit": value["memory_limit_exceeded"]})
    stage_table = markdown_table(pd.DataFrame(resources), ["stage", "minutes", "rss", "over_limit"],
                                ["Этап", "Минуты", "Peak tree RSS, МиБ", "Превышен лимит"])
    selected = pd.read_csv(directory / "selected_features.csv").astype(object).fillna("—")
    selected_table = markdown_table(selected.loc[selected["set"] == "selected"], ["feature", "manual_semantic_equivalent"],
                                    ["Выбранный агрегат", "Семантический аналог в M"])
    contrast_table = markdown_table(pd.DataFrame([{"contrast": name, **value} for name, value in paired.items()]),
                                    ["contrast", "mean", "lower", "upper"], ["Сравнение", "ΔNDCG", "95% lower", "95% upper"])
    prefix_table = markdown_table(prefixes, ["size", "ndcg"], ["Число A", "Внутренняя NDCG@10"])
    segments = pd.read_csv(directory / "test_segments.csv")
    # Keep the original segment schema and every value; no new subgroup inference.
    segment_table = markdown_table(segments, list(segments.columns))
    text = f"""# KION: ручные и автоматически построенные признаки рекомендаций

## Результат исследования

Построен воспроизводимый двухступенчатый рекомендатель: исторические источники
дают до 200 кандидатов, CatBoostRanker выбирает top-10. Полный baseline с 82
ручными входными колонками достигает NDCG@10 **{baseline['test_ndcg']:.5f}**,
на **{baseline['relative_gain_percent']:.2f}%** выше заранее выбранной популярности за 7 дней.
Во второй части реализованы генератор реляционных временных агрегатов и их
отбор на отдельном внутреннем временном разбиении. 16 автоматических агрегатов
заменяют 54 ручных агрегата; полный вход уменьшается с 82 до 44 колонок.

Автоматический вариант получил **{test.loc['auto', 'ndcg@10']:.5f}** на тесте.
Первичное Δ auto − full = **{primary['mean']:+.8f}**, 95% CI
**[{primary['lower']:+.8f}; {primary['upper']:+.8f}]**. Превосходство не подтверждено;
равенство и неухудшение также не доказаны: граница неухудшения заранее не задавалась.
Один заранее фиксированный случайный набор такого же размера получил
**{test.loc['random', 'ndcg@10']:.5f}** и превзошёл выбранный набор в диагностическом сравнении.
Это содержательный отрицательный результат для конкретного способа отбора,
одного seed и одной недели. После теста метод не менялся.
Диагностика после теста на внутреннем разбиении, без тестовых данных: среди
{subsets['protocol']['draws']} новых случайных наборов того же размера выбранный занимает
**{subset_rank['rank']}-е место из {subset_rank['of']}**, frozen random — {subset_frozen_rank['rank']}-е.
Отбор отсекает худшие наборы, но польза сверх случайного выбора не показана.

## 1. Данные и смысл задачи

Источник — [KION_DATASET](https://github.com/irsafilo/KION_DATASET), ревизия
`f69775be31fa5779907cf0a92ddedb70037fb5ae`; хеш архива и подготовленной выборки
находится в [sample_manifest.json](results/sample_manifest.json).
Из 962 179 пользователей равномерно выбраны 100 000 (seed=42), сохранена вся
их доступная история: 572 621 строка, 10 082 фильма. Исходный релиз содержит
5 476 251 строку и 15 706 фильмов; фактические даты — 13 марта — 22 августа 2021.

В KION одна строка соответствует паре пользователь–фильм с **последней** датой
просмотра. Это не полный журнал повторных просмотров: ранние просмотры пары,
последняя дата которой попала в будущее, восстановить нельзя. Предсказывается
наличие записи пары в следующей неделе, независимо от watched_pct. Метка 0
обозначает отсутствие записи, а не известный отказ после показа. Лога показов нет.
Статические метаданные считаются доступными на дату среза; их версий во времени нет.

Оценка условна на warm-active пользователях: есть историческая запись и хотя бы
одна целевая запись следующей недели. Это задача ранжирования для активной
аудитории, не оценка удержания, cold start или качества на всей базе.
На тесте 11 425 таких пользователей; 5 870 cold-active пользователей исключены.
873 цели тёплых пользователей относятся к новым фильмам и остаются в знаменателях.

## 2. Временной протокол и baseline

| Срез | Будущие метки | Роль |
|---|---|---|
| 19 июля 2021 | 19–25 июля | Обучение baseline; обучение внутреннего поиска |
| 26 июля | 26 июля – 1 августа | Обучение baseline; обучение внутреннего поиска |
| 2 августа | 2–8 августа | Обучение baseline; оценка внутреннего поиска |
| 9 августа | 9–15 августа | Внешняя валидация всех вариантов |
| 16 августа | 16–22 августа | Итоговый тест |

В каждый срез история ограничена строго датами `< cutoff`. Кандидаты и признаки
строятся до присоединения labels. До 5 000 обучающих пользователей случайно
выбираются на дату, но агрегаты используют историю всей фиксированной когорты.
Группы без найденного позитива исключены только из обучения; в оценке остаются.
Валидационное обучение: 2 223 979 пар / 11 120 запросов. Финальное обучение
включает также срез 9 августа: 2 940 779 пар / 14 704 запроса.

Источники: общая популярность, популярность за 7 дней, ItemKNN и жанровый профиль.
Каждый даёт до 100 кандидатов; сохраняются до 10 от каждого, остальное заполняет
RRF с константой 60 до общего лимита 200. Уже наблюдавшиеся пары исключаются;
будущие позитивы принудительно не добавляются. ItemKNN использует бинарную
историю до 100 последних фильмов, cosine × co/(co+10), до 100 соседей на фильм.
Все варианты ранжируют одинаковые упорядоченные пары и полные цели.

CatBoostRanker: YetiRank, depth=8, L2=3, learning_rate=0.05, seed=42,
max_ctr_complexity=1. Параметры baseline выбраны из сетки depth={{6,8}}, L2={{3,10}}
на внешней валидации, бюджет до 300 деревьев; для сравнений зафиксированы 299.
Финальные переобучения используют четыре прошлых среза без eval_set/early stopping.
В начальной сетке одна конфигурация переиспользовала проверенные первые 300 деревьев
старого запуска (1500 деревьев, 6 потоков); выбранная модель и финальные запуски
обучались заново в два потока. Происхождение отражено в
[grid_provenance.json](results/grid_provenance.json).

Внутренняя NDCG CatBoost нормируется по позитивам в кандидатах и используется
для early stopping baseline; внешний выбор конфигурации использует полные цели.
При нулевых позитивных кандидатах внешняя NDCG равна нулю, хотя внутренняя
метрика CatBoost 1.2.10 для такой группы возвращает 1.

## 3. Гипотезы и контролируемые группы признаков

| Семейство | Размер | Содержание |
|---|---:|---|
| S | 13 | Баллы, ранги, индикаторы четырёх источников и RRF |
| R | 15 | Статические поля пользователей/фильмов, ID, числовые поля и missing flags |
| M | 54 | Ручные исторические агрегаты пользователей, фильмов и связей пары |
| A | 16 | Автоматически сгенерированные и отобранные исторические агрегаты |
| random | 16 | Один случайный набор из той же грамматики; фактически без пересечений с A |

Первичный вопрос: меняется ли NDCG при замене M на A при фиксированных S и R?
Контроль S+R отделяет вклад новых агрегатов от статических данных. Сравнение
с random проверяет пользу специального отбора сверх самого словаря агрегатов.
Размер random, seed=42 и способ выбора зафиксированы до теста.

Диагностика без S на валидации: R+M (69 колонок) получила 0.14108188,
R+A (31 колонка) — 0.14086861. Кандидаты при этом остаются персональными;
это не система без рекомендательных источников. Ранняя абляция с 45 колонками
в первом отчёте исключает другую группу полей и не подменяет этот контроль.
Часть S и все источники кандидатов тоже спроектированы вручную: автоматизируется
построение семейства агрегатов M, а не весь рекомендательный конвейер.

## 4. Генерация и отбор

Идея обхода связей и применения агрегирующих операций опирается на
[Deep Feature Synthesis (Kanter, Veeramachaneni, 2015)](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf).
[OpenFE (Zhang et al., ICML 2023)](https://proceedings.mlr.press/v202/zhang23ay.html)
предлагает оценивать дополнительную пользу признаков через feature boosting
и сокращать кандидатов в два этапа. Здесь заимствована мотивация ограничить
пространство и стоимость поиска; реализован более простой importance-based
screen с временной оценкой префиксов. Feature boosting OpenFE и рекурсивный
синтез произвольной глубины DFS не реализованы. Полученные числа не являются
репликацией результатов этих статей или их оценкой на KION.

Грамматика задаёт 504 признака без просмотра меток. 10 путей: история пользователя
и фильма; история пользователя по жанру/режиссёру/стране/типу кандидата; история
фильма по возрасту/доходу/полу/kids_flg текущего пользователя. Окна: 3, 7, 10, 14,
30, 60, 90 дней и вся история. Пять операций (count, sum/mean log1p(duration),
mean/max watched_pct) дают 400 признаков, recency — 10, share — 64,
trend для пар окон 3/14, 7/30, 10/60 — 30.

`trend=(count(short)+1)/(count(long)+1)`. Share — отношение совпавшего
взвешенного числа записей к родительскому. Пустые count/sum равны 0,
mean/max/recency — NaN; share с пустым знаменателем равен 0. Для жанров
историческая запись распределяется с весом 1/число жанров, проекция на жанры
кандидата усредняется; mean имеет знаменатель только по непустым значениям.
Max/recency используют любое совпадение. Для режиссёра/страны берётся первый
элемент по той же нормализации, что у baseline. Это явные ограничения грамматики.

Для внутреннего поиска выбраны 1000 целых пользовательских запросов на дату,
история когорты сохранена. Обучение на 19/26 июля (400 тыс. пар), оценка на
2 августа (200 тыс. пар); 243 запроса без позитивов в пуле остаются в оценке.
На внутреннем обучении удалены два точных дубликата S: item count за 7 дней
и всю историю. Осталось 502 признака. Screen-модель (120 деревьев, depth=6)
ранжирует их по PredictionValuesChange **на обучении**. Это ранжирование
важности, не отдельная проверка качества. Префиксы 16/32/64 оцениваются
фиксированным ranker на внутреннем отложенном срезе:

{prefix_table}

Выбран минимальный префикс в пределах 0.0001 от лучшего: 16. Случайный контроль
выбран равномерно без возвращения из 502 допустимых признаков; в данном наборе
пересечений с A не оказалось. Внешняя валидация только сравнивает
семь фиксированных вариантов; метод затем запечатан в
[frozen.json](results/autofeatures/main/frozen.json), seal `{evidence['frozen']['seal']}`.
Правила заранее описаны в [протоколе](docs/automatic_features_protocol.md).

![Внутренний выбор размера](results/autofeatures/main/report/inner_selection.png)

## 5. Качество и интерпретация

{quality_table}

Метрики macro по пользователям. Recall использует полное число будущих целей;
MAP@10 делится на min(10, число целей); идеальная DCG также использует полный
целевой набор. На тесте общие 2 284 830 пар кандидатов. Candidate Recall@200 =
**0.54366744**; у 3 123 пользователей (27.33%) ни одной цели в пуле, у всех
ранжировщиков метрики для них равны нулю. Oracle NDCG при идеальном ранжировании
этого пула — **{baseline['oracle_candidate_ndcg']:.5f}**: улучшать можно и retrieval, и порядок.

Полный baseline выше популярности за 7 дней (0.13655864): ΔNDCG=+0.02073581,
95% CI [+0.01803895; +0.02355715]. Дополнительная старая абляция full−S на
валидации даёт +0.00379420, но включает и R, и M. Новый S+R показывает,
что нельзя целиком приписывать этот выигрыш ручным историческим агрегатам.

Парные разности тестовой NDCG:

{contrast_table}

Percentile bootstrap: 1000 повторов одних и тех же пользователей, seed=42.
Первичное сравнение — auto−full. Остальные диагностические, без поправки на
множественные сравнения. Интервалы условны на обученных моделях и одной неделе,
не учитывают вариацию переобучения, seed, выборки или следующего периода.

![Качество и интервалы](results/autofeatures/main/report/test_comparison.png)

Автогенерация даёт конкурентную точечную оценку при меньшем числе входных полей.
Однако дополнительная NDCG сверх S+R на тесте не подтверждена: соответствующий
интервал пересекает ноль. Прибавка auto над S включает также влияние R.
Выбранный набор проиграл одному случайному контролю по NDCG и MAP в
диагностических интервалах. Это не доказывает универсальную пользу случайного
поиска. У random самая высокая точечная NDCG среди пяти вариантов; первичная
гипотеза не заменяется на random−full после просмотра теста.

На внешней валидации auto−full=-0.00021504, CI [-0.00163593; +0.00118246],
auto−random=+0.00005740 с интервалом, включающим ноль. Валидационный почти
нулевой разрыв не обещал преимущества отбора на следующей неделе.

### Что выбрано и что из этого следует

{selected_table}

В наборе 8 сумм log-duration, 4 trend, 3 count и 1 share; 6 агрегатов используют
3 дня, 4 — отношение 10/60, 3 — 30 дней, 2 — 14 дней, 1 — всю историю.
Часто выбраны статистики фильма в целом и по демографии, затем жанровые
предпочтения пользователя. Отсутствие режиссёров в A означает результат
данного отбора, а не доказанную бесполезность этого пути.
Только `user_genres share 30` непосредственно совпадает с ручной формулой;
проверка на 10 000 реальных пар дала max abs error 5.96e-8. Вся грамматика
покрывает 43 из 54 ручных формул по смыслу, но не missing flags, active_days,
разнообразие и observed_age; средние и пропуски могут иметь другую семантику.
Полные определения A и random — в [CSV](results/autofeatures/main/selected_features.csv).

Доля A в PredictionValuesChange валидационной модели — 48.25%, S — 47.36%,
R — 4.38%. При коррелированных колонках важность распределяется между ними;
это не причинный эффект и не 48% дополнительного качества.
Возможные объяснения проигрыша random: коррелированные краткосрочные статистики
получили высокую обучающую важность, внутренние 1000 пользователей и одна неделя
дали нестабильный отбор. Более разнообразный random мог оказаться удачным.
Компонентных абляций здесь нет; насколько типичен сам случайный контроль,
проверено повторными наборами ниже.

### Повторные случайные наборы: диагностика после теста

Чтобы отличить слабый отбор от неудачного единственного сравнения, после теста
обучены ещё {subsets['protocol']['draws']} случайных наборов по {subsets['protocol']['size']} агрегатов
по правилу frozen random: равномерно без возвращения из {subsets['protocol']['eligible']}
допустимых признаков, с отдельными seed. Использовано только внутреннее
разбиение поиска: обучение 19/26 июля, оценка 2 августа на тех же 1000
пользователях. Тест не использовался; метод, frozen-печать и все прежние
результаты не менялись. Разбиение восстановлено из кэшей поиска и совпало
с fit выбранного префикса по SHA-256, повтор этого fit дал те же прогнозы
(atol=1e-10). Чтобы отделить вклад признаков от шума обучения, выбранный
набор обучен ещё с {subsets['protocol']['extra_seeds_for_selected']} seed CatBoost. Модели и параметры
те же: {subsets['protocol']['parameters']['iterations']} деревьев, depth={subsets['protocol']['parameters']['depth']}, без early stopping.

{subset_table}

![Выбранный набор среди случайных](results/autofeatures/main/report/random_subsets.png)

Выбранный набор — **{subset_rank['rank']}-й из {subset_rank['of']}** по NDCG@10:
{subset_rank['draws_at_least']} из {subsets['protocol']['draws']} случайных наборов не хуже,
одностороннее эмпирическое p = {subset_rank['empirical_p_one_sided']:.2f}. Он выше среднего
случайного набора на {subset_mean['mean']:+.5f}, парный 95% CI
[{subset_mean['lower']:+.5f}; {subset_mean['upper']:+.5f}] включает ноль. Из
{subsets['protocol']['draws']} парных интервалов {subset_excluded['selected_better']} исключают ноль в пользу выбранного
(против самых слабых наборов), в пользу случайного — {subset_excluded['random_better'] or 'ни один'}.

Разброс между случайными наборами (sd {subset_ndcg['random_std']:.5f}, от
{subset_ndcg['random_min']:.5f} до {subset_ndcg['random_max']:.5f}) примерно в {subset_ratio:.0f} раз больше
разброса выбранного набора по seed (sd {subset_ndcg['selected_seed_std']:.5f}). Значит, выбор
агрегатов действительно влияет на качество, но importance-based отбор
извлекает из этого мало: он избегает худших наборов, но не находит лучшие.
Случайные наборы почти не пересекаются с выбранным ({subset_overlap.min()}–{subset_overlap.max()} общих
агрегата из {subsets['protocol']['size']}), то есть близкое качество дают совсем другие признаки:
словарь избыточен, и сильные наборы в нём не единственны.

Frozen random и на внутреннем разбиении выше выбранного:
{subset_ndcg['frozen_random']:.5f}, {subset_frozen_rank['rank']}-е место из {subset_frozen_rank['of']}. Тестовым контролем
оказался удачный, а не типичный случайный набор. Остальные наборы на тесте
не оценивались, поэтому прямого вывода о тесте отсюда нет.

Оговорки: 1000 пользователей и одна неделя дают парные интервалы порядка
±0.005, шире большинства различий. Размер префикса 16 выбирался на этой же
оценке, что слегка завышает выбранный набор и только усиливает вывод.
Это дополнение после теста: оно объясняет результат, но не даёт нового
независимого подтверждения и не меняет первичный вывод. Детали —
[random_subsets.json](results/autofeatures/main/random_subsets.json),
[CSV](results/autofeatures/main/random_subsets.csv).

### Сегменты

{segment_table}

Это описательные оценки: отдельных доверительных интервалов и проверки гипотез
по сегментам нет. Нельзя объявлять победителя для конкретной аудитории по этим точкам.

## 6. Вычислительные затраты и масштабирование

Измерения на Apple M4, 16 GB RAM, Python 3.14.7, CatBoost 1.2.10; версии всех
зависимостей зафиксированы в requirements.lock. Тяжёлые запуски последовательны,
ограничены двумя вычислительными потоками и 6 GiB суммарного RSS дерева процессов.

{stage_table}

Итого поиск, внешняя валидация и финальный этап заняли около
{sum(float(r['minutes']) for r in resources):.1f} измеренных минут, без первого baseline,
пилота и последующих проверок. Это время сохранённых запусков, не обещание
длительности на другом компьютере и не календарная длительность с паузами сна.
Пилот на 1000 пользователях занял 22.30 с / 863.9 МиБ, с существенно меньшим
бюджетом модели; линейно переносить его время на основной эксперимент нельзя.

{cost_table}

Predict — медиана трёх прогретых пакетных вызовов на готовой тестовой матрице,
2.285 млн пар; retrieval, генерация признаков и Pool в него не входят.
Для полного baseline модель переиспользована, первоначальный fit занял 1049.38 с
вместе с Pool. Этот исторический замер не является синхронным сравнением fit.
RSS вариантов внутри одного рабочего процесса зависит от состояния аллокатора;
сравнивать его как изолированный расход памяти моделей нельзя.

Полная грамматика генерировалась на трёх внутренних матрицах по 200 тыс. пар:
17–19 с на матрицу. Объединение 32 колонок A/random для внешней валидации
строилось за 5.4 с на каждый обучающий срез и 14.64 с на 2.17 млн оценочных пар
(около 54 МиБ parquet). Эти расходы добавляются к уже построенным кандидатам
и S/R. Количество входов уменьшилось на 46.3%, но сериализованный auto даже
чуть больше full: число деревьев и структура сохраняются. Уменьшение числа
полей не доказывает меньшую память или ускорение всего конвейера.

Грамматика конечна, расчёты используют исторические группировки и проекции
на батчи пар (20 тыс.), а не полный user×item. Рост пользователей/фильмов,
жанровых разверток, числа окон и числа candidate-пар увеличивает память и
стоимость. До масштабирования нужны замеры на нескольких размерах, отдельные
процессные benchmark и инкрементальные агрегаты/хранилище признаков. Здесь
не измерены serving latency, QPS и распределённая обработка, не заявлена
готовность для миллиардов событий.

## 7. Надёжность, ограничения и соответствие заданию

Выполнены обе практические части: сильный classical ML baseline, собственная
реализация автоматических признаков, временной внутренний поиск, контроли,
сопоставимые кандидаты/пользователи/цели, зафиксированный финальный тест,
проверки утечки, повторная загрузка моделей, затраты и воспроизведение.
Это реализация и исследование автоматизации признаков, без заявления о новом
универсальном алгоритме или научной новизне самой идеи агрегирования.

Все 12 финальных/валидационных моделей проверены загрузкой, повторными
прогнозами и метриками (atol=1e-10). Схемы, параметры, 299 деревьев, порядок
пар и checksum артефактов проверены; исходные файлы baseline сохранены.
Повторный вызов финального этапа занял 1.03 с без изменения моделей и fit receipts.
Выполненный [ноутбук](notebooks/kion_research.ipynb) запрещает fit, проверяет
все модели, пересчитывает агрегаты на реальной истории для одного примера
и проверяет неизменность результата при искажении будущих значений.
Автотесты отдельно проверяют синтетические граничные случаи и утечки.

Ограничения: один датасет, одна когорта, seed и тестовая неделя; один random на тесте;
last-watch формат, отсутствие экспозиций и настоящих негативов; статические
метаданные; warm-active оценка; заранее известный тестовый результат baseline.
Правила новых моделей зафиксированы до их тестовой оценки, но это слабее
полностью скрытого от исследователя holdout. Bootstrap по пользователям не
исправляет эти ограничения и не доказывает онлайн-эффект.

Содержательный вывод: словарь автоматически построенных агрегатов способен
дать близкую точечную оценку, но выбранная схема importance-based отбора
не доказала пользу сверх random и не подтвердила прибавку NDCG к S+R на тесте.
На внутреннем разбиении она находится в середине распределения случайных
наборов того же размера.
Дальнейшие проверки — несколько прошлых временных folds, распределение
случайных контролей на внешней валидации, отбор с учётом корреляции и
стоимости — требуют нового протокола
и нового отложенного периода. Повторное использование этого теста для настройки
превратит его в development-набор. Новых вариантов после теста здесь не обучали.

## 8. Как проверить и воспроизвести

### Быстрая проверка готовых артефактов, без обучения

```bash
source .venv/bin/activate
python -m pytest -q
python -m kion_autofeatures verify
python -m scripts.build_research_report
python -m scripts.build_research_notebook --execute
```

Диагностика случайных наборов обучает модели и отдельно запускается командой
`python -m scripts.random_subsets` (около часа на данной машине, только внутреннее
разбиение основного run; готовые fit переиспользуются).

Ноутбук запускает полную проверку моделей; это десятки секунд на данной машине.
Генератор останавливается при отсутствии готовых артефактов, обучения по fallback нет.
Монитор RSS включает kernel и завершает собственное дерево процессов при превышении.
Запуск kernel требует разрешения на локальные соединения и наблюдение процессов.

### Полное воспроизведение на новой машине

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python main.py download
python main.py prepare
python main.py run
python -m scripts.verify_validation
python -m scripts.freeze_validation
python main.py lock-test
python main.py test
python -m scripts.feature_ablation
python -m kion_autofeatures reproduce --name reproduction_01
```

Численные версии и Python входят в identity. Данные, модели и кэши не хранятся
в Git; для быстрой проверки их нужно воспроизвести или перенести с сохранением
checksum. `reproduce` повторяет вторую часть с новым именем, включая подбор
внутри прошлого времени; это проверка воспроизводимости, а не новый независимый
holdout. Для другого имени отчёты стадий поддерживают `--name`; итоговый отчёт
и ноутбук по умолчанию используют основной run из autofeatures.toml.
Исходные замороженные результаты не перезаписывать. Изменённая ревизия кода
или окружение требуют нового согласованного набора артефактов.

Подробности команд и продолжения после прерывания:
[инструкция](docs/automatic_features_usage.md). Отчёт первой части сохранён
в [REPORT.md](REPORT.md), выполненный baseline-ноутбук —
[kion_baseline.ipynb](notebooks/kion_baseline.ipynb).
Промежуточные записи исследования: [валидация](results/autofeatures/main/VALIDATION.md),
[тест](results/autofeatures/main/TEST.md),
[аудит теста](results/autofeatures/main/test_verification.json).
Машинная сводка этого отчёта: [summary.json](results/autofeatures/main/report/summary.json).
"""
    (ROOT / "RESEARCH_REPORT.md").write_text(text)
    source_files = [directory / f"{name}.json" for name in evidence]
    source_files += [directory / name for name in ("validation_results.csv", "test_results.csv", "test_segments.csv",
                                                   "selected_features.csv", "random_subsets.csv")]
    source_files += [ROOT / "results" / "report_summary.json", Path(__file__)]
    summary = {"frozen_seal": evidence["frozen"]["seal"], "source_files": {
        str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "quality": quality, "primary_auto_minus_full": primary,
        "selected_count": len(selection["selected"]), "fit_called": False,
        "report_sha256": sha256(ROOT / "RESEARCH_REPORT.md"),
        "figures": {p.name: sha256(p) for p in sorted(output.glob("*.png"))}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return ROOT / "RESEARCH_REPORT.md"


if __name__ == "__main__":
    print(build())
