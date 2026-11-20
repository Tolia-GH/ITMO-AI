# ============================================================
# LimeSurvey Survey Visualization
# 自动分析问卷 CSV 并生成多种统计图表
# ============================================================

import os
import re
import html
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore")

# ============================================================
# 1. Configuration
# ============================================================

CSV_PATH = "./survey_results.csv"

OUTPUT_DIR = "./survey_charts"

# 图表语言
# 如果使用俄语标签，建议使用支持 Cyrillic 的字体
FONT_CANDIDATES = [
    "Arial"
]

plt.rcParams["font.family"] = FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid", context="notebook")

DPI = 300

# ============================================================
# 2. Data cleaning
# ============================================================

def clean_text(value):
    """
    清理单个单元格中的 HTML、转义字符和无效内容。

    处理：
    - <p>...</p>
    - <br>
    - &ndash;
    - &amp;
    - HTML entities
    - 空值
    - Н/Д
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    # HTML entity decoding
    value = html.unescape(value)

    # Remove HTML tags
    value = BeautifulSoup(value, "html.parser").get_text(
        separator=" ",
        strip=True
    )

    # Decode entities again in case some were nested
    value = html.unescape(value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    # Normalize missing values
    missing_values = {
        "Н/Д",
        "N/A",
        "NA",
        "NaN",
        "None",
        "null",
        "-"
    }

    if value in missing_values:
        return np.nan

    return value


def clean_dataframe(df):
    """
    对整个 DataFrame 进行清洗。
    """

    df.columns = [
        clean_text(col)
        for col in df.columns
    ]

    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    return df


# ============================================================
# 3. Load CSV
# ============================================================

def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    # LimeSurvey usually exports UTF-8 CSV
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str
        )

    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            dtype=str
        )

    df = clean_dataframe(df)

    print("=" * 60)
    print("Survey dataset loaded successfully")
    print("=" * 60)

    print(f"Number of respondents: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    return df


# ============================================================
# 4. Helper functions
# ============================================================

def save_figure(fig, folder, filename):

    target_folder = os.path.join(
        OUTPUT_DIR,
        folder
    )

    os.makedirs(target_folder, exist_ok=True)

    filepath = os.path.join(
        target_folder,
        filename
    )

    fig.savefig(
        filepath,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Saved: {filepath}")


def wrap_labels(labels, width=25):

    return [
        textwrap.fill(str(label), width=width)
        for label in labels
    ]


def get_question_columns(df, question_prefix):

    return [
        col
        for col in df.columns
        if col.startswith(question_prefix)
    ]


def extract_option(column_name):

    match = re.search(
        r"\[(.*?)\]",
        column_name
    )

    if match:
        return match.group(1)

    return column_name


def yes_no_to_bool(series):

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["да", "yes", "1", "true"])
    )


def valid_responses(series):

    return series.dropna()


def category_counts(series):

    return (
        series.dropna()
        .value_counts()
    )


def percentage(series):

    counts = category_counts(series)

    total = counts.sum()

    if total == 0:
        return counts

    return counts / total * 100


# ============================================================
# 5. Chart functions
# ============================================================

def plot_bar(
    series,
    title,
    xlabel,
    ylabel,
    folder,
    filename,
    horizontal=True,
    show_percentage=False
):

    counts = category_counts(series)

    if counts.empty:
        print(f"Skipped empty chart: {title}")
        return

    if horizontal:

        counts = counts.sort_values()

        fig, ax = plt.subplots(
            figsize=(10, max(5, len(counts) * 0.45))
        )

        bars = ax.barh(
            wrap_labels(counts.index, 35),
            counts.values
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    else:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        bars = ax.bar(
            wrap_labels(counts.index, 20),
            counts.values
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.xticks(rotation=35, ha="right")

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    # Add values to bars
    for bar in bars:

        width = bar.get_width()
        height = bar.get_height()

        if horizontal:

            ax.text(
                width + max(counts.values) * 0.01,
                bar.get_y() + height / 2,
                f"{int(width)}",
                va="center"
            )

        else:

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

    plt.tight_layout()

    save_figure(
        fig,
        folder,
        filename
    )


def plot_pie(
    series,
    title,
    folder,
    filename
):

    counts = category_counts(series)

    if counts.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=wrap_labels(counts.index, 20),
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        pctdistance=0.75
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    save_figure(
        fig,
        folder,
        filename
    )


def plot_histogram(
    series,
    title,
    xlabel,
    folder,
    filename,
    bins=10
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if values.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.histplot(
        values,
        bins=bins,
        kde=True,
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of respondents")

    plt.tight_layout()

    save_figure(
        fig,
        folder,
        filename
    )


def plot_boxplot(
    df,
    numeric_column,
    category_column,
    title,
    folder,
    filename
):

    temp = df[
        [numeric_column, category_column]
    ].copy()

    temp[numeric_column] = pd.to_numeric(
        temp[numeric_column],
        errors="coerce"
    )

    temp = temp.dropna()

    if temp.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.boxplot(
        data=temp,
        x=category_column,
        y=numeric_column,
        ax=ax
    )

    sns.stripplot(
        data=temp,
        x=category_column,
        y=numeric_column,
        color="black",
        alpha=0.4,
        jitter=True,
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(category_column)
    ax.set_ylabel(numeric_column)

    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    save_figure(
        fig,
        folder,
        filename
    )


# ============================================================
# 6. Demographic analysis
# ============================================================

def analyze_demographics(df):

    print("\n[1] Demographics")

    age_column = "Сколько вам лет?"

    status_column = (
        "Вы сейчас работаете или учитесь?"
    )

    # Age distribution
    if age_column in df.columns:

        age = pd.to_numeric(
            df[age_column],
            errors="coerce"
        )

        plot_histogram(
            age,
            title="Age Distribution",
            xlabel="Age",
            folder="01_demographics",
            filename="age_distribution.png",
            bins=10
        )

        # Age boxplot
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.boxplot(
            x=age,
            ax=ax
        )

        sns.stripplot(
            x=age,
            color="black",
            alpha=0.5,
            jitter=True,
            ax=ax
        )

        ax.set_title(
            "Age Distribution: Boxplot",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel("Age")

        plt.tight_layout()

        save_figure(
            fig,
            "01_demographics",
            "age_boxplot.png"
        )

    # Occupation
    if status_column in df.columns:

        plot_bar(
            df[status_column],
            title="Employment / Study Status",
            xlabel="Number of respondents",
            ylabel="Status",
            folder="01_demographics",
            filename="occupation_distribution.png"
        )

        plot_pie(
            df[status_column],
            title="Employment / Study Status",
            folder="01_demographics",
            filename="occupation_pie.png"
        )

        # Age grouped by status
        if age_column in df.columns:

            plot_boxplot(
                df,
                numeric_column=age_column,
                category_column=status_column,
                title="Age Distribution by Employment Status",
                folder="01_demographics",
                filename="age_by_status_boxplot.png"
            )


# ============================================================
# 7. Time distribution analysis
# ============================================================

def analyze_time_distribution(df):

    print("\n[2] Time distribution")

    work_column = (
        "Сколько часов в день вы обычно заняты "
        "учёбой или работой?"
    )

    free_column = (
        "Сколько свободного времени у вас обычно "
        "есть в день?"
    )

    if work_column in df.columns:

        plot_bar(
            df[work_column],
            title="Daily Study / Work Time",
            xlabel="Number of respondents",
            ylabel="Daily hours",
            folder="02_time_distribution",
            filename="work_study_hours.png",
            horizontal=False
        )

        plot_pie(
            df[work_column],
            title="Daily Study / Work Time",
            folder="02_time_distribution",
            filename="work_study_hours_pie.png"
        )

    if free_column in df.columns:

        plot_bar(
            df[free_column],
            title="Daily Free Time",
            xlabel="Number of respondents",
            ylabel="Free time",
            folder="02_time_distribution",
            filename="free_time_distribution.png",
            horizontal=False
        )

        plot_pie(
            df[free_column],
            title="Daily Free Time",
            folder="02_time_distribution",
            filename="free_time_pie.png"
        )


# ============================================================
# 8. Multiple-choice activity analysis
# ============================================================

def analyze_multiple_choice(
    df,
    question_prefix,
    title,
    folder,
    filename
):

    columns = get_question_columns(
        df,
        question_prefix
    )

    if not columns:
        print(f"No columns found: {question_prefix}")
        return None

    results = []

    for col in columns:

        option = extract_option(col)

        # "Other (selected)" is a text field,
        # not a Yes/No checkbox
        if option.lower() == "other (selected)":
            continue

        selected = yes_no_to_bool(
            df[col]
        )

        count = selected.sum()

        valid_n = df[col].notna().sum()

        pct = (
            count / valid_n * 100
            if valid_n > 0
            else 0
        )

        results.append({
            "Activity": option,
            "Count": count,
            "Percentage": pct
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return None

    result_df = result_df.sort_values(
        "Count",
        ascending=True
    )

    # Horizontal bar chart
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(result_df) * 0.45))
    )

    bars = ax.barh(
        wrap_labels(
            result_df["Activity"],
            width=35
        ),
        result_df["Count"]
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Number of respondents"
    )

    ax.set_ylabel(
        "Activity"
    )

    # Show count and percentage
    for i, bar in enumerate(bars):

        count = result_df.iloc[i]["Count"]
        pct = result_df.iloc[i]["Percentage"]

        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({pct:.1f}%)",
            va="center"
        )

    plt.tight_layout()

    save_figure(
        fig,
        folder,
        filename
    )

    return result_df


# ============================================================
# 9. Actual leisure activities
# ============================================================

def analyze_actual_activities(df):

    print("\n[3] Actual leisure activities")

    prefix = (
        "Как вы обычно проводите свободное время? "
    )

    return analyze_multiple_choice(
        df,
        question_prefix=prefix,
        title="How Respondents Actually Spend Their Free Time",
        folder="03_actual_activities",
        filename="actual_activities.png"
    )


# ============================================================
# 10. Satisfaction analysis
# ============================================================

def analyze_satisfaction(df):

    print("\n[4] Satisfaction")

    satisfaction_column = (
        "Довольны ли вы своим свободным временем?"
    )

    sufficient_column = (
        "Хватает ли вам свободного времени?"
    )

    if satisfaction_column in df.columns:

        plot_pie(
            df[satisfaction_column],
            title="Satisfaction with Free Time",
            folder="04_satisfaction",
            filename="satisfaction_pie.png"
        )

        plot_bar(
            df[satisfaction_column],
            title="Satisfaction with Free Time",
            xlabel="Number of respondents",
            ylabel="Satisfaction",
            folder="04_satisfaction",
            filename="satisfaction_bar.png"
        )

    if sufficient_column in df.columns:

        plot_pie(
            df[sufficient_column],
            title="Is Free Time Sufficient?",
            folder="04_satisfaction",
            filename="free_time_sufficiency_pie.png"
        )

        plot_bar(
            df[sufficient_column],
            title="Is Free Time Sufficient?",
            xlabel="Number of respondents",
            ylabel="Response",
            folder="04_satisfaction",
            filename="free_time_sufficiency_bar.png"
        )


# ============================================================
# 11. Ideal leisure activities
# ============================================================

def analyze_ideal_activities(df):

    print("\n[5] Ideal leisure activities")

    prefix = (
        "Как вы бы хотели проводить своё свободное время? "
    )

    return analyze_multiple_choice(
        df,
        question_prefix=prefix,
        title="How Respondents Would Like to Spend Their Free Time",
        folder="05_ideal_activities",
        filename="ideal_activities.png"
    )


# ============================================================
# 12. Barriers analysis
# ============================================================

def analyze_barriers(df):

    print("\n[6] Barriers")

    prefix = (
        "Какой фактор чаще всего мешает вам "
        "проводить свободное время по желанию? "
    )

    return analyze_multiple_choice(
        df,
        question_prefix=prefix,
        title="Factors Preventing Respondents from Spending Free Time as Desired",
        folder="06_barriers",
        filename="barriers.png"
    )


# ============================================================
# 13. Actual vs Ideal activities
# ============================================================

def analyze_actual_vs_ideal(
    df,
    actual_df,
    ideal_df
):

    print("\n[7] Actual vs Ideal")

    if actual_df is None or ideal_df is None:
        return

    comparison = pd.merge(
        actual_df,
        ideal_df,
        on="Activity",
        how="outer",
        suffixes=("_Actual", "_Ideal")
    )

    comparison = comparison.fillna(0)

    comparison = comparison.sort_values(
        "Percentage_Ideal",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    y = np.arange(
        len(comparison)
    )

    height = 0.38

    ax.barh(
        y - height / 2,
        comparison["Percentage_Actual"],
        height,
        label="Actual"
    )

    ax.barh(
        y + height / 2,
        comparison["Percentage_Ideal"],
        height,
        label="Ideal"
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        wrap_labels(
            comparison["Activity"],
            35
        )
    )

    ax.set_xlabel(
        "Percentage of respondents (%)"
    )

    ax.set_title(
        "Actual vs Ideal Leisure Activities",
        fontsize=15,
        fontweight="bold"
    )

    ax.legend()

    ax.xaxis.set_major_formatter(
        PercentFormatter(100)
    )

    plt.tight_layout()

    save_figure(
        fig,
        "07_comparison",
        "actual_vs_ideal.png"
    )

    # Calculate the gap
    comparison["Gap"] = (
        comparison["Percentage_Ideal"]
        - comparison["Percentage_Actual"]
    )

    comparison = comparison.sort_values(
        "Gap",
        ascending=False
    )

    # Gap chart
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    colors = [
        "tab:blue" if x >= 0 else "tab:red"
        for x in comparison["Gap"]
    ]

    ax.barh(
        wrap_labels(
            comparison["Activity"],
            35
        ),
        comparison["Gap"],
        color=colors
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xlabel(
        "Ideal - Actual (%)"
    )

    ax.set_title(
        "Difference Between Desired and Actual Activities",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    save_figure(
        fig,
        "07_comparison",
        "activity_gap.png"
    )

    # Export comparison statistics
    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "actual_vs_ideal_statistics.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nActual vs Ideal statistics:")
    print(comparison)


# ============================================================
# 14. Export descriptive statistics
# ============================================================

def export_statistics(df):

    print("\n[8] Export descriptive statistics")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    statistics = {}

    for col in df.columns:

        counts = category_counts(
            df[col]
        )

        if not counts.empty:

            statistics[col] = counts

    with open(
        os.path.join(
            OUTPUT_DIR,
            "descriptive_statistics.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Survey Descriptive Statistics\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Total respondents: {len(df)}\n\n"
        )

        for col, counts in statistics.items():

            f.write(
                f"\nQuestion: {col}\n"
            )

            f.write(
                "-" * 40 + "\n"
            )

            total = counts.sum()

            for category, count in counts.items():

                pct = count / total * 100

                f.write(
                    f"{category}: "
                    f"{count} "
                    f"({pct:.2f}%)\n"
                )

    print(
        "Descriptive statistics exported."
    )


# ============================================================
# 15. Main
# ============================================================

def main():

    df = load_data(
        CSV_PATH
    )

    # Create output folder
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # 1. Demographics
    analyze_demographics(df)

    # 2. Time distribution
    analyze_time_distribution(df)

    # 3. Actual activities
    actual_df = analyze_actual_activities(df)

    # 4. Satisfaction
    analyze_satisfaction(df)

    # 5. Ideal activities
    ideal_df = analyze_ideal_activities(df)

    # 6. Barriers
    analyze_barriers(df)

    # 7. Actual vs ideal
    analyze_actual_vs_ideal(
        df,
        actual_df,
        ideal_df
    )

    # 8. Statistics
    export_statistics(df)

    print("\n" + "=" * 60)

    print("All charts generated successfully!")

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()