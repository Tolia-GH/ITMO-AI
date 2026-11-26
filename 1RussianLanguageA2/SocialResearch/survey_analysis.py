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
    return [c for c in df.columns
            if c.startswith(prefix) and not c.endswith("[Other (selected)]")]


def yes_count(series):
    return series.astype(str).str.strip().str.lower().isin(YES).sum()


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


def plot_multi_select(df, columns, title, filename, charts_dir, pdf):
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
def report_distribution(series, question_name):
    counts, pcts, total = count_percent(series)
    if total == 0:
        return f"По вопросу «{question_name}» нет валидных ответов."
    top = counts.index[0]
    return (f"На вопрос «{question_name}» ответили {total} {russian_people(total)}. "
            f"Наиболее распространённый ответ — «{top}» "
            f"({counts.iloc[0]} {russian_people(int(counts.iloc[0]))}, "
            f"{fmt_pct(pcts.iloc[0])}%).")


def multi_summary(table, intro):
    if table.empty:
        return intro + " Данные отсутствуют."
    order = np.argsort(-table["Процент"].to_numpy(), kind="stable")
    top = table.iloc[order[:3]]
    pieces = [
        f"«{r['Вариант']}» — {int(r['Количество'])} "
        f"{russian_people(int(r['Количество']))} ({fmt_pct(r['Процент'])}%)"
        for _, r in top.iterrows()
    ]
    return intro + " Наиболее популярные варианты: " + "; ".join(pieces) + "."


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
         int(row["Фактически, ответивших"]), int(row["Хотели бы, n"]),
         f"{fmt_pct(row['Хотели бы, %'])}%", int(row["Хотели бы, ответивших"])]
        for _, row in table.iterrows()
    ]
    return markdown_table(
        ["Занятие", "Фактически, чел.", "Фактически, %", "Ответивших, чел.",
         "Хотели бы, чел.", "Хотели бы, %", "Ответивших, чел."], rows
    )


def _legacy_main():
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

    chart_files = []
    summary_tables = {}
    age_section = None
    employment_section = None
    daily_sections = []
    constraints_sections = []
    actual_table = pd.DataFrame()
    satisfaction_counts = pd.Series(dtype=int)
    barrier_table = pd.DataFrame()
    comparison_table = None

    def categorical_table(series):
        counts, percentages, total = count_percent(series)
        rows = [
            [answer, int(count), f"{fmt_pct(percentages.loc[answer])}%"]
            for answer, count in counts.items()
        ]
        rendered = markdown_table(
            ["Вариант ответа", "Количество, чел.", "Доля, %"], rows
        )
        return counts, percentages, total, rendered

    def multi_respondents(columns):
        if not columns:
            return 0
        valid = df[columns].apply(
            lambda column: column.astype(str).str.strip().str.lower().isin(YES | NO)
        )
        return int(valid.any(axis=1).sum())

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
                    f"{age_table}\n\nВсего валидных ответов: **{len(ages)}**.\n\n"
                    "![Возраст — диаграмма размаха](charts/age_boxplot.png)\n\n"
                    f"{age_table}\n\nВсего валидных ответов: **{len(ages)}**."
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
                _, _, total, table = categorical_table(df[status_col])
                employment_section = (
                    "### Вы сейчас работаете или учитесь?*\n\n"
                    f"{report_distribution(df[status_col], 'статус занятости')}\n\n"
                    "![Статус занятости](charts/employment_status.png)\n\n"
                    f"{table}\n\nВсего валидных ответов: **{total}**."
                )

        for col, title, filename, question, caption in [
            (workload_col, "Ежедневная занятость учёбой / работой", "workload.png",
             "Сколько часов в день вы обычно заняты учёбой или работой?",
             "Ежедневная занятость учёбой / работой"),
            (free_col, "Количество свободного времени в день", "free_time.png",
             "Сколько свободного времени у вас обычно есть в день?",
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
                    _, _, total, table = categorical_table(df[col])
                    daily_sections.append(
                        f"### {question}*\n\n{report_distribution(df[col], question)}\n\n"
                        f"![{caption}](charts/{filename})\n\n{table}\n\n"
                        f"Всего валидных ответов: **{total}**."
                    )

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
                    f"На вопрос ответили **{multi_respondents(actual_cols)} человек**.\n\n"
                    f"{multi_summary(actual_table, 'Респонденты могли выбрать несколько вариантов.')}\n\n"
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
                satisfaction_counts, _, total, table = categorical_table(df[satisfaction_col])
                constraints_sections.append(
                    "### Довольны ли вы своим свободным временем?*\n\n"
                    f"{report_distribution(df[satisfaction_col], 'Довольны ли вы своим свободным временем?')}\n\n"
                    "![Удовлетворённость свободным временем](charts/satisfaction.png)\n\n"
                    f"{table}\n\nВсего валидных ответов: **{total}**."
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
                _, _, total, table = categorical_table(df[enough_col])
                constraints_sections.append(
                    "### Хватает ли вам свободного времени?\n\n"
                    "*Вопрос открыт только для тех, кто ответил «Не доволен(а) своим свободным временем».*\n\n"
                    f"{report_distribution(df[enough_col], 'Хватает ли вам свободного времени?')}\n\n"
                    "![Достаточность свободного времени](charts/enough_free_time.png)\n\n"
                    f"{table}\n\nВсего валидных ответов: **{total}**."
                )

        if desired_cols:
            path, table = plot_multi_select(
                df, desired_cols, "Желаемые способы проведения свободного времени",
                "desired_activities.png", charts_dir, pdf
            )
            if path:
                chart_files.append(path)
                summary_tables["Желаемые занятия"] = table
                constraints_sections.append(
                    "### Как вы бы хотели проводить своё свободное время? (Не больше 5 выборов.)\n\n"
                    "*Вопрос открыт только для тех, кто ответил «Не доволен(а) своим свободным временем».*\n\n"
                    f"На вопрос ответили **{multi_respondents(desired_cols)} человек**.\n\n"
                    f"{multi_summary(table, 'Респонденты могли выбрать несколько вариантов.')}\n\n"
                    "![Желаемые занятия](charts/desired_activities.png)\n\n"
                    f"{multi_select_markdown_table(table)}"
                )

        if actual_cols and desired_cols:
            path, comparison_table = plot_actual_vs_desired(
                df, actual_cols, desired_cols, charts_dir, pdf
            )
            if path:
                chart_files.append(path)
                summary_tables["Фактическое и желаемое"] = comparison_table
                comparison_table["Разница, п.п."] = (
                    comparison_table["Хотели бы, %"] - comparison_table["Фактически, %"]
                )
                top_gaps = comparison_table.sort_values(
                    "Разница, п.п.", ascending=False, kind="stable"
                ).head(3)
                gap_text = "; ".join(
                    f"«{row['Вариант']}» (+{fmt_pct(row['Разница, п.п.'])} п.п.)"
                    for _, row in top_gaps.iterrows()
                )
                constraints_sections.append(
                    "### Сравнение фактических и желаемых занятий\n\n"
                    f"Наиболее выраженный разрыв между желаемым и фактическим участием наблюдается "
                    f"в следующих занятиях: {gap_text}.\n\n"
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
                constraints_sections.append(
                    "### Какой фактор чаще всего мешает вам проводить свободное время по желанию? "
                    "(Не больше 2 выборов.)\n\n"
                    "*Вопрос открыт только для тех, кто ответил «Не доволен(а) своим свободным временем».*\n\n"
                    f"На вопрос ответили **{multi_respondents(obstacle_cols)} человек**.\n\n"
                    f"{multi_summary(barrier_table, 'Можно было выбрать несколько факторов.')}\n\n"
                    "![Ограничивающие факторы](charts/barriers.png)\n\n"
                    f"{multi_select_markdown_table(barrier_table)}"
                )

    df.to_csv(out_dir / "cleaned_data.csv", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(out_dir / "summary_tables.xlsx", engine="openpyxl") as writer:
            for name, table in summary_tables.items():
                safe_sheet = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
                table.to_excel(writer, sheet_name=safe_sheet, index=False)
    except ImportError:
        print("openpyxl не установлен; Excel-файл не создан (установите: pip install openpyxl).")

    activity_text = "Нет валидных ответов."
    if not actual_table.empty:
        top_activity = actual_table.sort_values(
            "Процент", ascending=False, kind="stable"
        ).iloc[0]
        count = int(top_activity["Количество"])
        activity_text = (
            f"«{top_activity['Вариант']}» ({count} {russian_people(count)}, "
            f"{fmt_pct(top_activity['Процент'])}%)."
        )

    satisfaction_text = "Нет валидных ответов."
    if not satisfaction_counts.empty:
        answer = satisfaction_counts.index[0]
        count = int(satisfaction_counts.iloc[0])
        total = int(satisfaction_counts.sum())
        satisfaction_text = (
            f"«{answer}» ({count} {russian_people(count)} из {total}, "
            f"{fmt_pct(count / total * 100)}%)."
        )

    barrier_text = "Нет валидных ответов."
    if not barrier_table.empty:
        top_barrier = barrier_table.sort_values(
            "Процент", ascending=False, kind="stable"
        ).iloc[0]
        count = int(top_barrier["Количество"])
        barrier_text = (
            f"«{top_barrier['Вариант']}» ({count} {russian_people(count)}, "
            f"{fmt_pct(top_barrier['Процент'])}%)."
        )

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
    for section in constraints_sections:
        report.extend([section, "", "---", ""])
    report.extend([
        "## Общие выводы",
        "",
        f"Чаще всего респонденты проводят свободное время так: {activity_text}",
        f"В вопросе об удовлетворённости наиболее распространённый ответ — {satisfaction_text}",
        f"Наиболее часто отмечаемое ограничение — {barrier_text}",
        "Результаты описывают только участников этого опроса и не должны автоматически переноситься на всю молодёжь.",
    ])

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
