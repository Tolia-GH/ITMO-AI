# -*- coding: utf-8 -*-
"""
Автоматический анализ анкеты об организации свободного времени.
Запуск:
    python survey_analysis.py "responses.csv"
Или без аргумента — программа попросит путь к CSV.

Результаты сохраняются в папку survey_report рядом с CSV:
    charts/*.png
    survey_report.md
    survey_charts.pdf
    cleaned_data.csv
    summary_tables.xlsx (если установлен openpyxl)
"""

import sys
import re
import html
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# --------------------------- Настройки ---------------------------
RUSSIAN_FONT_CANDIDATES = [
    "Arial", "DejaVu Sans", "Liberation Sans", "Noto Sans",
    "Microsoft YaHei", "Arial Unicode MS"
]
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = RUSSIAN_FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220

YES = {"да", "yes", "1", "true"}
NO = {"нет", "no", "0", "false"}
NA_VALUES = {"", "н/д", "n/a", "na", "нет ответа", "nan", "none", "null"}

ACTUAL_PREFIX = "Как вы обычно проводите свободное время? "
DESIRED_PREFIX = "Как вы бы хотели проводить своё свободное время? "
OBSTACLE_PREFIX = "Какой фактор чаще всего мешает вам проводить свободное время по желанию? "

# --------------------------- Очистка данных ---------------------------
def clean_text(value):
    """Удаляет HTML, декодирует HTML entities и приводит текст к нормальному виду."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)                 # &ndash; -> –, &amp; -> &
    text = re.sub(r"<[^>]+>", " ", text)       # <p>...</p>, <br>, etc.
    text = re.sub(r"\r\n|\n|\r|\t", " ", text)
    text = text.replace("\\\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer(value):
    value = clean_text(value)
    if value.lower() in NA_VALUES:
        return np.nan
    return value


def read_csv_robust(path):
    """Пробует распространённые кодировки и разделители."""
    encodings = ["utf-8-sig", "utf-8", "cp1251"]
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            if len(df.columns) > 1:
                return df
        except Exception as exc:
            last_error = exc
    # Fallback: LimeSurvey обычно экспортирует CSV с запятыми и кавычками.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"Не удалось прочитать CSV: {last_error or exc}")


def clean_dataframe(df):
    df.columns = [clean_text(c) for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(normalize_answer)
    if "Возраст" in df.columns:
        df["Возраст"] = pd.to_numeric(df["Возраст"], errors="coerce")
    age_col = next((c for c in df.columns if c.startswith("Сколько вам лет?")), None)
    if age_col:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    return df


def option_from_column(column, prefix):
    """Из названия вопроса с [вариант] извлекает вариант ответа."""
    raw = column[len(prefix):].strip() if column.startswith(prefix) else column
    match = re.search(r"\[(.*?)\]", raw)
    return clean_text(match.group(1)) if match else clean_text(raw)


def find_columns(df, prefix):
    ignored_options = ("[Other (selected)]", "[Другое]")
    return [
        c for c in df.columns
        if c.startswith(prefix) and not c.endswith(ignored_options)
    ]


def valid_responses(series):
    return series.dropna().astype(str).map(clean_text).replace("", np.nan).dropna()


def count_percent(series):
    s = valid_responses(series)
    counts = s.value_counts()
    total = counts.sum()
    return counts, (counts / total * 100 if total else counts.astype(float)), total


def fmt_pct(x):
    return f"{x:.1f}".replace(".", ",")


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def russian_people(count):
    last_two = count % 100
    last = count % 10
    if 11 <= last_two <= 14:
        return "человек"
    if last == 1:
        return "человек"
    if 2 <= last <= 4:
        return "человека"
    return "человек"


def save_fig(fig, filename, charts_dir, pdf):
    fig.tight_layout()
    path = charts_dir / filename
    fig.savefig(path, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_categorical(series, title, filename, charts_dir, pdf, kind="bar", horizontal=False):
    counts, percentages, total = count_percent(series)
    if not total:
        return None, counts, percentages, total
    labels = counts.index.astype(str).tolist()
    values = counts.values
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)) if horizontal else (9, 6))
    if kind == "pie":
        ax.pie(values, labels=labels, autopct=lambda p: f"{p:.1f}%", startangle=90,
               counterclock=False, wedgeprops={"linewidth": 1, "edgecolor": "white"})
        ax.set_title(title, pad=20)
        ax.axis("equal")
    elif horizontal:
        bars = ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel("Количество респондентов")
        ax.set_title(title)
        for bar, count in zip(bars, values[::-1]):
            pct = count / total * 100
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
                    f"{count} ({fmt_pct(pct)}%)", va="center", fontsize=10)
        ax.set_xlim(0, max(values) * 1.22)
    else:
        bars = ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel("Количество респондентов")
        ax.tick_params(axis="x", rotation=25)
        for bar, count in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f"{count} ({fmt_pct(count/total*100)}%)", ha="center", fontsize=10)
    path = save_fig(fig, filename, charts_dir, pdf)
    return path, counts, percentages, total


def plot_multi_select(df, columns, title, filename, charts_dir, pdf, make_plot=True):
    """Для вопросов с ответами Да/Нет: число и процент выбравших каждый вариант."""
    rows = []
    for col in columns:
        option = option_from_column(col, ACTUAL_PREFIX if col.startswith(ACTUAL_PREFIX)
                                    else DESIRED_PREFIX if col.startswith(DESIRED_PREFIX)
                                    else OBSTACLE_PREFIX)
        series = df[col]
        valid = series.astype(str).str.strip().str.lower().isin(YES | NO)
        n_valid = int(valid.sum())
        n_yes = int(series.astype(str).str.strip().str.lower().isin(YES).sum())
        if n_valid:
            rows.append({
                "Вариант": option,
                "Количество": n_yes,
                "Процент": n_yes / n_valid * 100,
                "Ответивших": n_valid,
            })
    if not rows:
        return None, pd.DataFrame(columns=["Вариант", "Количество", "Процент"])
    table = pd.DataFrame(rows).sort_values("Количество", kind="stable").reset_index(drop=True)
    if not make_plot:
        return None, table
    labels = table["Вариант"].tolist()
    counts = table["Количество"].tolist()
    pcts = table["Процент"].tolist()
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.5)))
    bars = ax.barh(labels, counts)
    ax.set_title(title)
    ax.set_xlabel("Количество ответов «Да»")
    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count * 1.25 + 1)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_width() + max_count * 0.025, bar.get_y()+bar.get_height()/2,
                f"{count} ({fmt_pct(pct)}%)", va="center", fontsize=10)
    path = save_fig(fig, filename, charts_dir, pdf)
    return path, table


def plot_actual_vs_desired(df, actual_cols, desired_cols, charts_dir, pdf):
    actual_map = {option_from_column(c, ACTUAL_PREFIX): c for c in actual_cols}
    desired_map = {option_from_column(c, DESIRED_PREFIX): c for c in desired_cols}
    options = [x for x in actual_map if x in desired_map]
    rows = []
    for option in options:
        actual = df[actual_map[option]]
        desired = df[desired_map[option]]
        actual_valid = actual.astype(str).str.lower().isin(YES | NO)
        desired_valid = desired.astype(str).str.lower().isin(YES | NO)
        na = int(actual.astype(str).str.lower().isin(YES).sum())
        nd = int(desired.astype(str).str.lower().isin(YES).sum())
        da = int(actual_valid.sum())
        dd = int(desired_valid.sum())
        if not da or not dd:
            continue
        rows.append({
            "Вариант": option,
            "Фактически, %": na / da * 100 if da else np.nan,
            "Хотели бы, %": nd / dd * 100 if dd else np.nan,
            "Фактически, n": na, "Хотели бы, n": nd,
            "Фактически, ответивших": da, "Хотели бы, ответивших": dd,
        })
    table = pd.DataFrame(rows).sort_values("Хотели бы, %", ascending=True, kind="stable")
    if table.empty:
        return None, table
    fig, ax = plt.subplots(figsize=(12, max(7, len(table)*0.55)))
    y = np.arange(len(table))
    h = 0.36
    ax.barh(y-h/2, table["Фактически, %"], h, label="Фактические занятия")
    ax.barh(y+h/2, table["Хотели бы, %"], h, label="Желаемые занятия")
    ax.set_yticks(y, table["Вариант"])
    ax.set_xlabel("Доля респондентов, %")
    ax.set_title("Фактические и желаемые способы проведения досуга")
    ax.legend()
    ax.set_xlim(0, max(100, table[["Фактически, %", "Хотели бы, %"]].max().max()*1.12))
    path = save_fig(fig, "actual_vs_desired.png", charts_dir, pdf)
    return path, table


def plot_age(df, age_col, charts_dir, pdf):
    age = pd.to_numeric(df[age_col], errors="coerce").dropna()
    if age.empty:
        return None, age
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = max(5, min(10, int(age.nunique())))
    sns.histplot(age, bins=bins, kde=(len(age) >= 5), ax=ax)
    ax.set_title("Распределение респондентов по возрасту")
    ax.set_xlabel("Возраст, лет")
    ax.set_ylabel("Количество респондентов")
    save_fig(fig, "age_histogram.png", charts_dir, pdf)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.boxplot(x=age, ax=ax)
    ax.set_title("Возраст: диаграмма размаха")
    ax.set_xlabel("Возраст, лет")
    path = save_fig(fig, "age_boxplot.png", charts_dir, pdf)
    return path, age


# --------------------------- Автоматический текст отчёта ---------------------------
def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def multi_select_markdown_table(table):
    rows = [
        [row["Вариант"], int(row["Количество"]), f"{fmt_pct(row['Процент'])}%",
         int(row["Ответивших"])]
        for _, row in table.iterrows()
    ]
    return markdown_table(
        ["Вариант ответа", "Выбрали, чел.", "Доля, %", "Ответивших, чел."], rows
    )


def comparison_markdown_table(table):
    rows = [
        [row["Вариант"], int(row["Фактически, n"]), f"{fmt_pct(row['Фактически, %'])}%",
         int(row["Хотели бы, n"]), f"{fmt_pct(row['Хотели бы, %'])}%"]
        for _, row in table.iterrows()
    ]
    return markdown_table(
        ["Занятие", "Фактически, чел.", "Фактически, %",
         "Хотели бы, чел.", "Хотели бы, %"], rows
    )


def question_summary(series, strip_answer_period=False):
    counts, percentages, total = count_percent(series)
    if not total:
        return "На вопрос ответов нет.", total
    answer = str(counts.index[0])
    if strip_answer_period:
        answer = answer.rstrip(".")
    count = int(counts.iloc[0])
    text = (
        f"На вопрос ответили **{total} {russian_people(total)}**. \n\n"
        f"Наиболее распространённый ответ — «{answer}» "
        f"({count} {russian_people(count)}, {fmt_pct(percentages.iloc[0])}%)."
    )
    return text, total


def popular_bullets(table, blank_after_heading=True):
    order = np.argsort(-table["Процент"].to_numpy(), kind="stable")[:3]
    top = table.iloc[order]
    lines = [
        f"- «{row['Вариант']}» — {int(row['Количество'])} "
        f"{russian_people(int(row['Количество']))} ({fmt_pct(row['Процент'])}%);"
        for _, row in top.iterrows()
    ]
    if lines:
        lines[-1] = lines[-1][:-1] + "."
    separator = "\n\n" if blank_after_heading else "\n"
    return "Наиболее популярные варианты: " + separator + "\n".join(lines)


def gap_bullets(table):
    gaps = table.assign(
        **{"Разница, п.п.": table["Хотели бы, %"] - table["Фактически, %"]}
    ).sort_values("Разница, п.п.", ascending=False, kind="stable").head(3)
    lines = [
        f"- «{row['Вариант']}» (+{fmt_pct(row['Разница, п.п.'])} п.п.);"
        for _, row in gaps.iterrows()
    ]
    if lines:
        lines[-1] = lines[-1][:-1] + "."
    return (
        "Наиболее выраженный разрыв между желаемым и фактическим участием наблюдается "
        "в следующих занятиях: \n" + "\n".join(lines)
    )


def main():
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1]).expanduser()
    else:
        csv_path = Path(input("Введите путь к CSV-файлу: ").strip().strip('"'))
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    df = clean_dataframe(read_csv_robust(csv_path))
    n = len(df)
    out_dir = csv_path.parent / "survey_report"
    charts_dir = out_dir / "charts"
    out_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)

    age_col = next((c for c in df.columns if c.startswith("Сколько вам лет?")), None)
    status_col = next((c for c in df.columns if c.startswith("Вы сейчас работаете или учитесь?")), None)
    workload_col = next((c for c in df.columns if c.startswith(
        "Сколько часов в день вы обычно заняты учёбой или работой?")), None)
    free_col = next((c for c in df.columns if c.startswith(
        "Сколько свободного времени у вас обычно есть в день?")), None)
    satisfaction_col = next((c for c in df.columns if c.startswith(
        "Довольны ли вы своим свободным временем?")), None)
    enough_col = next((c for c in df.columns if c.startswith(
        "Хватает ли вам свободного времени?")), None)
    actual_cols = find_columns(df, ACTUAL_PREFIX)
    desired_cols = find_columns(df, DESIRED_PREFIX)
    obstacle_cols = find_columns(df, OBSTACLE_PREFIX)

    # Compare activities only for dissatisfied respondents with complete answers.
    activity_cols = actual_cols + desired_cols
    if satisfaction_col and actual_cols and desired_cols:
        valid_activity_answers = df[activity_cols].apply(
            lambda column: column.astype(str).str.strip().str.lower().isin(YES | NO)
        )
        dissatisfied = df[satisfaction_col].astype(str).str.strip().str.lower().eq("нет")
        activity_df = df.loc[dissatisfied & valid_activity_answers.all(axis=1)].copy()
    else:
        activity_df = df.iloc[0:0].copy()

    chart_files = []
    summary_tables = {}

    def categorical_table(series, preferred_order=None):
        counts, percentages, total = count_percent(series)
        if preferred_order:
            order = [answer for answer in preferred_order if answer in counts.index]
            order.extend(answer for answer in counts.index if answer not in order)
            counts = counts.reindex(order)
            percentages = percentages.reindex(order)
        rows = [
            [answer, int(count), f"{fmt_pct(percentages.loc[answer])}%"]
            for answer, count in counts.items()
        ]
        return markdown_table(["Вариант ответа", "Количество, чел.", "Доля, %"], rows)

    def single_section(series, question, image_alt, image_file, preferred_order=None,
                       strip_answer_period=False):
        intro, _ = question_summary(series, strip_answer_period=strip_answer_period)
        return (
            f"### {question}\n\n{intro}\n\n"
            f"![{image_alt}](charts/{image_file})\n\n"
            f"{categorical_table(series, preferred_order)}"
        )

    age_section = None
    employment_section = None
    daily_sections = []
    constraint_sections = []
    with PdfPages(out_dir / "survey_charts.pdf") as pdf:
        if age_col:
            age_path, ages = plot_age(df, age_col, charts_dir, pdf)
            if age_path:
                chart_files.extend([
                    charts_dir / "age_histogram.png",
                    charts_dir / "age_boxplot.png",
                ])
                age_counts = ages.value_counts().sort_index()
                age_rows = [
                    [int(age) if float(age).is_integer() else fmt_num(age),
                     int(count), f"{fmt_pct(count / len(ages) * 100)}%"]
                    for age, count in age_counts.items()
                ]
                age_table = markdown_table(
                    ["Возраст, лет", "Количество, чел.", "Доля, %"], age_rows
                )
                age_section = (
                    "### Сколько вам лет?*\n\n"
                    f"На вопрос ответили **{len(ages)} человек**.\n\n"
                    f"Диапазон: **{int(ages.min())}–{int(ages.max())}** лет\n"
                    f"Средний возраст: **{fmt_num(ages.mean())}** года\n"
                    f"Медиана: **{fmt_num(ages.median())}**\n\n"
                    "![Возраст — гистограмма](charts/age_histogram.png)\n\n"
                    "![Возраст — диаграмма размаха](charts/age_boxplot.png)\n\n"
                    f"{age_table}"
                )

        if status_col:
            path, counts, percentages, total = plot_categorical(
                df[status_col], "Статус занятости респондентов",
                "employment_status.png", charts_dir, pdf, kind="pie"
            )
            if path:
                chart_files.append(path)
                summary_tables["Статус занятости"] = pd.DataFrame({
                    "Количество": counts, "Процент": percentages,
                })
                employment_section = single_section(
                    df[status_col], "Вы сейчас работаете или учитесь?*",
                    "Статус занятости", "employment_status.png"
                )

        for col, title, filename, question, caption in [
            (workload_col, "Ежедневная занятость учёбой / работой", "workload.png",
             "Сколько часов в день вы обычно заняты учёбой или работой?*",
             "Ежедневная занятость учёбой / работой"),
            (free_col, "Количество свободного времени в день", "free_time.png",
             "Сколько свободного времени у вас обычно есть в день?*",
             "Количество свободного времени в день"),
        ]:
            if col:
                path, counts, percentages, total = plot_categorical(
                    df[col], title, filename, charts_dir, pdf, horizontal=True
                )
                if path:
                    chart_files.append(path)
                    summary_tables[question] = pd.DataFrame({
                        "Количество": counts, "Процент": percentages,
                    })
                    preferred = None
                    if col == workload_col:
                        preferred = ["8–10 часов", "4–6 часов", "6–8 часов", "Более 10 часов"]
                    elif col == free_col:
                        preferred = [
                            "Более 5 часов", "3–5 часов", "2–3 часа",
                            "1–2 часа", "Менее 1 часа",
                        ]
                    intro, _ = question_summary(df[col])
                    daily_sections.append(
                        f"### {question}\n\n{intro}\n\n"
                        f"![{caption}](charts/{filename})\n\n"
                        f"{categorical_table(df[col], preferred)}"
                    )

        actual_table = pd.DataFrame()
        desired_table = pd.DataFrame()
        comparison_table = pd.DataFrame()
        if actual_cols:
            path, actual_table = plot_multi_select(
                df, actual_cols, "Как респонденты проводят свободное время",
                "actual_activities.png", charts_dir, pdf
            )
            if path:
                chart_files.append(path)
                summary_tables["Фактические занятия"] = actual_table
                daily_sections.append(
                    "### Как вы обычно проводите свободное время? (Не больше 5 выборов.)*\n\n"
                    f"На вопрос ответили **{n} человек**.\n\n"
                    f"{popular_bullets(actual_table)}\n\n"
                    "![Фактические занятия](charts/actual_activities.png)\n\n"
                    f"{multi_select_markdown_table(actual_table)}"
                )

        if satisfaction_col:
            path, counts, percentages, total = plot_categorical(
                df[satisfaction_col], "Удовлетворённость свободным временем",
                "satisfaction.png", charts_dir, pdf, kind="pie"
            )
            if path:
                chart_files.append(path)
                summary_tables["Удовлетворённость"] = pd.DataFrame({
                    "Количество": counts, "Процент": percentages,
                })
                constraint_sections.append(
                    single_section(
                        df[satisfaction_col], "Довольны ли вы своим свободным временем?*",
                        "Удовлетворённость свободным временем", "satisfaction.png"
                    )
                )

        if enough_col:
            path, counts, percentages, total = plot_categorical(
                df[enough_col], "Достаточность свободного времени",
                "enough_free_time.png", charts_dir, pdf, kind="pie"
            )
            if path:
                chart_files.append(path)
                summary_tables["Достаточность свободного времени"] = pd.DataFrame({
                    "Количество": counts, "Процент": percentages,
                })
                intro, _ = question_summary(df[enough_col], strip_answer_period=True)
                constraint_sections.append(
                    "### Хватает ли вам свободного времени?\n\n"
                    "*Вопрос открыт только для тех, кто ответил «Не доволен(а) своим свободным временем».*\n\n"
                    f"{intro}\n\n"
                    "![Достаточность свободного времени](charts/enough_free_time.png)\n\n"
                    f"{categorical_table(df[enough_col])}"
                )

        if desired_cols and not activity_df.empty:
            _, desired_table = plot_multi_select(
                activity_df, desired_cols, "Желаемые способы проведения свободного времени",
                "desired_activities.png", charts_dir, pdf, make_plot=False
            )
            summary_tables["Желаемые занятия"] = desired_table

        if actual_cols and desired_cols and not activity_df.empty:
            path, comparison_table = plot_actual_vs_desired(
                activity_df, actual_cols, desired_cols, charts_dir, pdf
            )
            if path:
                chart_files.append(path)
                summary_tables["Фактическое и желаемое"] = comparison_table
                desired_summary = (
                    f"На вопрос ответили **{len(activity_df)} человек**.\n\n"
                    f"{popular_bullets(desired_table, blank_after_heading=False)}"
                )
                constraint_sections.append(
                    "### Как вы бы хотели проводить своё свободное время? (Не больше 5 выборов.)\n\n"
                    "*Вопрос открыт только для тех, кто ответил «Не доволен(а) своим свободным временем».*\n\n"
                    f"{desired_summary}\n\n{gap_bullets(comparison_table)}\n\n"
                    "![Сравнение](charts/actual_vs_desired.png)\n\n"
                    f"{comparison_markdown_table(comparison_table)}"
                )

        if obstacle_cols:
            path, barrier_table = plot_multi_select(
                df, obstacle_cols, "Факторы, мешающие проводить досуг по желанию",
                "barriers.png", charts_dir, pdf
            )
            if path:
                chart_files.append(path)
                summary_tables["Ограничивающие факторы"] = barrier_table
                constraint_sections.append(
                    "### Какой фактор чаще всего мешает вам проводить свободное время по желанию? "
                    "(Не больше 2 выборов.)\n\n"
                    "*Вопрос открыт только для тех, кто ответил «Не доволен(а) своим свободным временем».*\n\n"
                    f"На вопрос ответили **{int(barrier_table['Ответивших'].max())} человек**.\n\n"
                    f"{popular_bullets(barrier_table, blank_after_heading=False)}\n\n"
                    "![Ограничивающие факторы](charts/barriers.png)\n\n"
                    f"{multi_select_markdown_table(barrier_table)}"
                )

    obsolete_desired_chart = charts_dir / "desired_activities.png"
    if obsolete_desired_chart.exists():
        obsolete_desired_chart.unlink()

    df.to_csv(out_dir / "cleaned_data.csv", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(out_dir / "summary_tables.xlsx", engine="openpyxl") as writer:
            for name, table in summary_tables.items():
                safe_sheet = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
                table.to_excel(writer, sheet_name=safe_sheet, index=False)
    except ImportError:
        print("openpyxl не установлен; Excel-файл не создан (установите: pip install openpyxl).")

    report = [
        '# Социологического опроса "Как молодые люди проводят свободное время?"',
        "",
        "**Цель исследования:** Изучить, как молодые люди проводят свободное время и насколько они удовлетворены его количеством.",
        "",
        "**Формат опроса:** Онлайн-анкета.",
        "",
        f"**Количество полученных анкет:** {n}.",
        "",
        "---",
        "",
        "## Общая информация",
        "",
    ]
    if age_section:
        report.extend([age_section, "", "---", ""])
    if employment_section:
        report.extend([employment_section, "", "---", ""])
    report.extend(["## Свободное время и повседневная жизнь", ""])
    for section in daily_sections:
        report.extend([section, "", "---", ""])
    report.extend(["## Ограничения и пожелания", ""])
    for index, section in enumerate(constraint_sections):
        report.extend([section])
        if index < len(constraint_sections) - 1:
            report.extend(["", "---", ""])

    report_path = out_dir / "survey_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nАнализ завершён.")
    print(f"Исходный CSV: {csv_path}")
    print(f"Результаты: {out_dir}")
    print(f"Отчёт: {report_path}")
    print(f"PDF с графиками: {out_dir / 'survey_charts.pdf'}")
    print(f"Всего графиков: {len(chart_files)}")


if __name__ == "__main__":
    main()
