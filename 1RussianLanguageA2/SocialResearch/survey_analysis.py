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
import unicodedata
from pathlib import Path
from collections import Counter

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

# Удаляем служебные/метаданные LimeSurvey из аналитики
META_COLUMNS = {
    "ID ответа", "Дата записи", "Последняя страница", "Начальный язык",
    "Зерно", "Дата начала", "Время последнего обращения"
}


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
    return [c for c in df.columns if c.startswith(prefix)]


def yes_count(series):
    return series.astype(str).str.strip().str.lower().isin(YES).sum()


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


def plot_multi_select(df, columns, title, filename, charts_dir, pdf):
    """Для вопросов с ответами Да/Нет: число и процент выбравших каждый вариант."""
    labels, counts, pcts = [], [], []
    for col in columns:
        option = option_from_column(col, ACTUAL_PREFIX if col.startswith(ACTUAL_PREFIX)
                                    else DESIRED_PREFIX if col.startswith(DESIRED_PREFIX)
                                    else OBSTACLE_PREFIX)
        series = df[col]
        valid = series.astype(str).str.strip().str.lower().isin(YES | NO)
        n_valid = int(valid.sum())
        n_yes = int(series.astype(str).str.strip().str.lower().isin(YES).sum())
        if n_valid:
            labels.append(option)
            counts.append(n_yes)
            pcts.append(n_yes / n_valid * 100)
    if not labels:
        return None, pd.DataFrame(columns=["Вариант", "Количество", "Процент"])
    order = np.argsort(counts)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    pcts = [pcts[i] for i in order]
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
    table = pd.DataFrame({"Вариант": labels, "Количество": counts, "Процент": pcts})
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
        rows.append({
            "Вариант": option,
            "Фактически, %": na / da * 100 if da else np.nan,
            "Хотели бы, %": nd / dd * 100 if dd else np.nan,
            "Фактически, n": na, "Хотели бы, n": nd
        })
    table = pd.DataFrame(rows).sort_values("Хотели бы, %", ascending=True)
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


def plot_age_by_free_time(df, age_col, free_col, charts_dir, pdf):
    if age_col not in df or free_col not in df:
        return None
    subset = df[[age_col, free_col]].dropna()
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    order = valid_responses(df[free_col]).value_counts().index.tolist()
    sns.boxplot(data=subset, x=free_col, y=age_col, order=order, ax=ax)
    ax.set_title("Возраст в зависимости от количества свободного времени")
    ax.set_xlabel("Свободное время в день")
    ax.set_ylabel("Возраст, лет")
    ax.tick_params(axis="x", rotation=20)
    return save_fig(fig, "age_by_free_time.png", charts_dir, pdf)


def plot_workload_satisfaction(df, workload_col, satisfaction_col, charts_dir, pdf):
    sub = df[[workload_col, satisfaction_col]].dropna()
    if sub.empty:
        return None
    tab = pd.crosstab(sub[workload_col], sub[satisfaction_col], normalize="index") * 100
    # Preserve natural category order as observed
    fig, ax = plt.subplots(figsize=(10, 6))
    tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Удовлетворённость досугом в зависимости от занятости")
    ax.set_ylabel("Доля внутри группы, %")
    ax.set_xlabel("Продолжительность учёбы / работы")
    ax.legend(title="Удовлетворённость", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=20)
    return save_fig(fig, "workload_vs_satisfaction.png", charts_dir, pdf)


# --------------------------- Автоматический текст отчёта ---------------------------
def report_distribution(series, question_name):
    counts, pcts, total = count_percent(series)
    if total == 0:
        return f"По вопросу «{question_name}» нет валидных ответов."
    top = counts.index[0]
    return (f"На вопрос «{question_name}» ответили {total} человек. "
            f"Наиболее распространённый ответ — «{top}» "
            f"({counts.iloc[0]} человек, {fmt_pct(pcts.iloc[0])}%).")


def multi_summary(table, intro):
    if table.empty:
        return intro + " Данные отсутствуют."
    top = table.sort_values("Процент", ascending=False).head(3)
    pieces = [f"«{r['Вариант']}» — {int(r['Количество'])} ({fmt_pct(r['Процент'])}%)"
              for _, r in top.iterrows()]
    return intro + " Наиболее популярные варианты: " + "; ".join(pieces) + "."


def main():
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1]).expanduser()
    else:
        csv_path = Path(input("Введите путь к CSV-файлу: ").strip().strip('"'))

    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    raw = read_csv_robust(csv_path)
    df = clean_dataframe(raw)
    n = len(df)
    out_dir = csv_path.parent / "survey_report"
    charts_dir = out_dir / "charts"
    out_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)

    # Выявляем основные колонки по тексту вопроса
    age_col = next((c for c in df.columns if c.startswith("Сколько вам лет?")), None)
    status_col = next((c for c in df.columns if c.startswith("Вы сейчас работаете или учитесь?")), None)
    workload_col = next((c for c in df.columns if c.startswith("Сколько часов в день вы обычно заняты учёбой или работой?")), None)
    free_col = next((c for c in df.columns if c.startswith("Сколько свободного времени у вас обычно есть в день?")), None)
    satisfaction_col = next((c for c in df.columns if c.startswith("Довольны ли вы своим свободным временем?")), None)
    enough_col = next((c for c in df.columns if c.startswith("Хватает ли вам свободного времени?")), None)

    actual_cols = find_columns(df, ACTUAL_PREFIX)
    desired_cols = find_columns(df, DESIRED_PREFIX)
    obstacle_cols = find_columns(df, OBSTACLE_PREFIX)

    chart_files = []
    report_sections = []
    summary_tables = {}

    with PdfPages(out_dir / "survey_charts.pdf") as pdf:
        # 1. Возраст
        if age_col:
            path, ages = plot_age(df, age_col, charts_dir, pdf)
            if path:
                chart_files += [charts_dir / "age_histogram.png", path]
                report_sections.append(
                    f"### Возраст участников\n\nВ опросе приняли участие **{len(ages)}** человек "
                    f"с указанным возрастом. Средний возраст — **{fmt_num(ages.mean())}** года, "
                    f"медиана — **{fmt_num(ages.median())}**, диапазон — **{int(ages.min())}–{int(ages.max())}** лет.\n\n"
                    f"![Возраст — гистограмма](charts/age_histogram.png)\n\n"
                    f"![Возраст — boxplot](charts/age_boxplot.png)"
                )

        # 2. Статус занятости
        if status_col:
            path, counts, pcts, total = plot_categorical(
                df[status_col], "Статус занятости респондентов", "employment_status.png",
                charts_dir, pdf, kind="pie")
            if path:
                chart_files.append(path)
                summary_tables["Статус занятости"] = pd.DataFrame({"Количество": counts, "Процент": pcts})
                report_sections.append(
                    f"### Работа и учёба\n\n{report_distribution(df[status_col], 'Статус занятости')}\n\n"
                    f"![Статус занятости](charts/employment_status.png)"
                )

        # 3. Время работы/учёбы и свободное время
        for col, title, filename, section in [
            (workload_col, "Ежедневная занятость учёбой / работой", "workload.png", "Продолжительность учёбы / работы"),
            (free_col, "Количество свободного времени в день", "free_time.png", "Количество свободного времени")
        ]:
            if col:
                path, counts, pcts, total = plot_categorical(df[col], title, filename, charts_dir, pdf, horizontal=True)
                if path:
                    chart_files.append(path)
                    summary_tables[section] = pd.DataFrame({"Количество": counts, "Процент": pcts})
                    report_sections.append(f"### {section}\n\n{report_distribution(df[col], section)}\n\n![{section}](charts/{filename})")

        # 4. Actual activities and desired activities
        if actual_cols:
            path, table = plot_multi_select(df, actual_cols, "Как респонденты проводят свободное время",
                                            "actual_activities.png", charts_dir, pdf)
            if path:
                chart_files.append(path)
                summary_tables["Фактические занятия"] = table
                report_sections.append("### Фактическое проведение свободного времени\n\n" +
                    multi_summary(table, "Респонденты могли выбрать несколько вариантов.") +
                    "\n\n![Фактические занятия](charts/actual_activities.png)")
        if desired_cols:
            path, table = plot_multi_select(df, desired_cols, "Желаемые способы проведения свободного времени",
                                            "desired_activities.png", charts_dir, pdf)
            if path:
                chart_files.append(path)
                summary_tables["Желаемые занятия"] = table
                report_sections.append("### Желаемые занятия\n\n" +
                    multi_summary(table, "Респонденты могли выбрать несколько вариантов.") +
                    "\n\n![Желаемые занятия](charts/desired_activities.png)")
        if actual_cols and desired_cols:
            path, table = plot_actual_vs_desired(df, actual_cols, desired_cols, charts_dir, pdf)
            if path:
                chart_files.append(path)
                summary_tables["Фактическое и желаемое"] = table
                # Наибольший разрыв между желаемым и фактическим
                table["Разница, п.п."] = table["Хотели бы, %"] - table["Фактически, %"]
                gap = table.sort_values("Разница, п.п.", ascending=False).head(3)
                gap_text = "; ".join(f"«{r['Вариант']}» (+{fmt_pct(r['Разница, п.п.'])} п.п.)"
                                     for _, r in gap.iterrows())
                report_sections.append(
                    "### Сравнение фактических и желаемых занятий\n\n"
                    f"Наиболее выраженный разрыв между желаемым и фактическим участием наблюдается в следующих занятиях: {gap_text}.\n\n"
                    "![Сравнение](charts/actual_vs_desired.png)"
                )

        # 5. Satisfaction and whether free time is sufficient
        for col, title, filename, section in [
            (satisfaction_col, "Удовлетворённость свободным временем", "satisfaction.png", "Удовлетворённость"),
            (enough_col, "Достаточность свободного времени", "enough_free_time.png", "Достаточность свободного времени")
        ]:
            if col:
                path, counts, pcts, total = plot_categorical(df[col], title, filename, charts_dir, pdf, kind="pie")
                if path:
                    chart_files.append(path)
                    summary_tables[section] = pd.DataFrame({"Количество": counts, "Процент": pcts})
                    report_sections.append(f"### {section}\n\n{report_distribution(df[col], section)}\n\n![{section}](charts/{filename})")

        # 6. Barriers
        if obstacle_cols:
            path, table = plot_multi_select(df, obstacle_cols, "Факторы, мешающие проводить досуг по желанию",
                                            "barriers.png", charts_dir, pdf)
            if path:
                chart_files.append(path)
                summary_tables["Ограничивающие факторы"] = table
                report_sections.append("### Факторы, ограничивающие свободное время\n\n" +
                    multi_summary(table, "Можно было выбрать несколько факторов.") +
                    "\n\n![Ограничивающие факторы](charts/barriers.png)")

        # 7. Exploratory charts
        if age_col and free_col:
            path = plot_age_by_free_time(df, age_col, free_col, charts_dir, pdf)
            if path:
                chart_files.append(path)
                report_sections.append("### Возраст и количество свободного времени\n\n"
                                      "Диаграмма позволяет визуально сравнить распределение возраста в группах с разным объёмом свободного времени. "
                                      "Она носит описательный характер и сама по себе не доказывает причинно-следственную связь.\n\n"
                                      "![Возраст и свободное время](charts/age_by_free_time.png)")
        if workload_col and satisfaction_col:
            path = plot_workload_satisfaction(df, workload_col, satisfaction_col, charts_dir, pdf)
            if path:
                chart_files.append(path)
                report_sections.append("### Занятость и удовлетворённость\n\n"
                                      "На диаграмме показана доля ответов об удовлетворённости внутри каждой категории ежедневной занятости. "
                                      "При небольшом числе участников сравнения следует интерпретировать осторожно.\n\n"
                                      "![Занятость и удовлетворённость](charts/workload_vs_satisfaction.png)")

    # Сохраняем очищенные данные и таблицы
    cleaned_path = out_dir / "cleaned_data.csv"
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(out_dir / "summary_tables.xlsx", engine="openpyxl") as writer:
            for name, table in summary_tables.items():
                safe_sheet = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
                table.to_excel(writer, sheet_name=safe_sheet, index=True if table.index.name is None and "Количество" in table.columns else False)
    except ImportError:
        print("openpyxl не установлен; Excel-файл не создан (установите: pip install openpyxl).")

    # Отчёт на русском
    report = [
        "# Отчёт по результатам социологического опроса",
        "",
        f"**Количество полученных анкет:** {n}.",
        "",
        "## Цель исследования",
        "",
        "Изучить, как респонденты проводят свободное время, насколько они удовлетворены его количеством, "
        "какие занятия хотели бы выбрать и какие обстоятельства мешают им реализовать свои предпочтения.",
        "",
        "## Краткая характеристика выборки",
        "",
        f"В анализ включено **{n}** анкет. При расчёте долей для каждого вопроса использовались только валидные ответы на соответствующий вопрос. "
        "В вопросах с множественным выбором процент рассчитан от числа ответивших на данный пункт; сумма процентов может превышать 100%.",
        "",
        *report_sections,
        "",
        "## Общие выводы",
        "",
        "Результаты позволяют сопоставить повседневные занятия респондентов с их предпочтениями, а также оценить роль учебной/рабочей нагрузки, "
        "времени на дорогу, финансовых и других ограничений. Выводы следует рассматривать как описание данной выборки, а не как оценку всей популяции.",
        "",
        "## Методологические примечания",
        "",
        "- Ответы `Н/Д`, пустые ячейки и технические значения исключались из расчёта соответствующего вопроса.",
        "- В вопросах с множественным выбором подсчитывались ответы «Да» по каждому варианту независимо.",
        "- Для категориальных вопросов использовались частоты и проценты; возраст анализировался как числовая переменная.",
        "- При небольшом объёме выборки не следует трактовать наблюдаемые различия как статистически значимые без отдельной проверки.",
        "",
        "## Файлы",
        "",
        "- `survey_charts.pdf` — все графики в одном PDF-файле.",
        "- `charts/` — отдельные изображения графиков.",
        "- `summary_tables.xlsx` — сводные таблицы (если доступен пакет `openpyxl`).",
        "- `cleaned_data.csv` — очищенная версия исходных данных.",
    ]
    report_path = out_dir / "survey_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nАнализ завершён.")
    print(f"Исходный CSV: {csv_path}")
    print(f"Результаты: {out_dir}")
    print(f"Отчёт: {report_path}")
    print(f"PDF с графиками: {out_dir / 'survey_charts.pdf'}")
    print(f"Всего графиков: {len(chart_files)}")


if __name__ == "__main__":
    main()
