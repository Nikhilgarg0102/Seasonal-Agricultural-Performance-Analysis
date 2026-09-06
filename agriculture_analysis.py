# ============================================================
# SEASONAL AGRICULTURE PERFORMANCE ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

FILE_PATH = "C:/Users/hp/OneDrive/Desktop/Project/seasonal_agriculture_performance_dataset.csv"

df = pd.read_csv(FILE_PATH)

print("\n" + "=" * 60)
print("SEASONAL AGRICULTURE PERFORMANCE ANALYSIS")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ------------------------------------------------------------
# 3. MISSING VALUES & DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill numerical missing values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# ------------------------------------------------------------
# 4. DISPLAY DATA TYPES
# ------------------------------------------------------------

print("\nNumerical Columns:")
print(list(numeric_columns))

print("\nCategorical Columns:")
print(list(categorical_columns))


# ------------------------------------------------------------
# 5. BASIC VISUALIZATION
# ------------------------------------------------------------

# Numerical distributions

if len(numeric_columns) > 0:

    df[numeric_columns].hist(
        figsize=(14, 10),
        bins=20
    )

    plt.suptitle(
        "Distribution of Agricultural Variables",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. SEASONAL ANALYSIS
# ------------------------------------------------------------

if "Season" in df.columns:

    print("\n" + "=" * 60)
    print("SEASONAL ANALYSIS")
    print("=" * 60)

    print("\nNumber of Records per Season:")
    print(df["Season"].value_counts())

    # Find performance column
    if "Yield" in df.columns:

        season_summary = df.groupby("Season")["Yield"].agg(
            ["count", "mean", "median", "min", "max", "std"]
        )

        print("\nSeason-wise Yield:")
        print(season_summary)

        # Visualization
        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=df,
            x="Season",
            y="Yield"
        )

        plt.title("Average Agricultural Yield by Season")
        plt.xlabel("Season")
        plt.ylabel("Average Yield")

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Boxplot
        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=df,
            x="Season",
            y="Yield"
        )

        plt.title("Yield Distribution Across Seasons")
        plt.xlabel("Season")
        plt.ylabel("Yield")

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# 7. PRODUCTION ANALYSIS
# ------------------------------------------------------------

if "Season" in df.columns and "Production" in df.columns:

    production = df.groupby("Season")["Production"].mean()

    print("\n" + "=" * 60)
    print("SEASONAL PRODUCTION")
    print("=" * 60)

    print(production)

    plt.figure(figsize=(10, 6))

    production.plot(kind="bar")

    plt.title("Average Production by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Production")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. REGION ANALYSIS
# ------------------------------------------------------------

if "Region" in df.columns and "Yield" in df.columns:

    region_performance = (
        df.groupby("Region")["Yield"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("REGION-WISE PERFORMANCE")
    print("=" * 60)

    print(region_performance)

    plt.figure(figsize=(12, 6))

    region_performance.plot(kind="bar")

    plt.title("Average Yield by Region")
    plt.xlabel("Region")
    plt.ylabel("Average Yield")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. CROP ANALYSIS
# ------------------------------------------------------------

if "Crop" in df.columns and "Yield" in df.columns:

    crop_performance = (
        df.groupby("Crop")["Yield"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("CROP-WISE PERFORMANCE")
    print("=" * 60)

    print(crop_performance)

    plt.figure(figsize=(12, 6))

    crop_performance.plot(kind="bar")

    plt.title("Average Yield by Crop")
    plt.xlabel("Crop")
    plt.ylabel("Average Yield")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 10. RAINFALL VS YIELD
# ------------------------------------------------------------

if (
    "Rainfall" in df.columns
    and "Yield" in df.columns
):

    plt.figure(figsize=(9, 6))

    if "Season" in df.columns:

        sns.scatterplot(
            data=df,
            x="Rainfall",
            y="Yield",
            hue="Season"
        )

    else:

        sns.scatterplot(
            data=df,
            x="Rainfall",
            y="Yield"
        )

    plt.title("Rainfall vs Agricultural Yield")
    plt.xlabel("Rainfall")
    plt.ylabel("Yield")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11. TEMPERATURE VS YIELD
# ------------------------------------------------------------

if (
    "Temperature" in df.columns
    and "Yield" in df.columns
):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="Temperature",
        y="Yield",
        hue="Season" if "Season" in df.columns else None
    )

    plt.title("Temperature vs Agricultural Yield")
    plt.xlabel("Temperature")
    plt.ylabel("Yield")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 12. RESOURCE USAGE
# ------------------------------------------------------------

resource_columns = [
    "Fertilizer",
    "Water",
    "Irrigation",
    "Pesticide"
]

available_resources = [
    column for column in resource_columns
    if column in df.columns
]

if "Season" in df.columns and available_resources:

    print("\n" + "=" * 60)
    print("RESOURCE USAGE BY SEASON")
    print("=" * 60)

    resource_summary = df.groupby("Season")[
        available_resources
    ].mean()

    print(resource_summary)

    resource_summary.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Average Resource Usage by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Usage")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 13. CORRELATION ANALYSIS
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] >= 2:

    correlation = numeric_df.corr()

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    print(correlation)

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Agricultural Variables Correlation Matrix")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 14. ANOVA TEST
# ------------------------------------------------------------

if (
    "Season" in df.columns
    and "Yield" in df.columns
):

    season_groups = [
        group["Yield"].dropna().values
        for _, group in df.groupby("Season")
    ]

    if len(season_groups) >= 2:

        f_stat, p_value = stats.f_oneway(
            *season_groups
        )

        print("\n" + "=" * 60)
        print("ANOVA TEST")
        print("=" * 60)

        print("F-statistic:", round(f_stat, 4))
        print("P-value:", round(p_value, 6))

        if p_value < 0.05:

            print(
                "Conclusion: Seasonal differences "
                "in yield are statistically significant."
            )

        else:

            print(
                "Conclusion: No statistically significant "
                "seasonal difference was detected."
            )


# ------------------------------------------------------------
# 15. ECONOMIC ANALYSIS
# ------------------------------------------------------------

if "Season" in df.columns:

    economic_columns = [
        "Revenue",
        "Cost",
        "Profit"
    ]

    available_economic = [
        column for column in economic_columns
        if column in df.columns
    ]

    if available_economic:

        print("\n" + "=" * 60)
        print("ECONOMIC PERFORMANCE")
        print("=" * 60)

        economic_summary = df.groupby("Season")[
            available_economic
        ].mean()

        print(economic_summary)

        economic_summary.plot(
            kind="bar",
            figsize=(12, 6)
        )

        plt.title("Economic Performance by Season")
        plt.xlabel("Season")
        plt.ylabel("Value")

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# 16. KEY FINDINGS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

if "Season" in df.columns and "Yield" in df.columns:

    season_yield = df.groupby("Season")["Yield"].mean()

    best_season = season_yield.idxmax()
    best_yield = season_yield.max()

    worst_season = season_yield.idxmin()
    worst_yield = season_yield.min()

    print(
        f"\nBest performing season: {best_season}"
    )

    print(
        f"Average yield: {best_yield:.2f}"
    )

    print(
        f"\nLowest performing season: {worst_season}"
    )

    print(
        f"Average yield: {worst_yield:.2f}"
    )


if "Region" in df.columns and "Yield" in df.columns:

    region_yield = df.groupby("Region")["Yield"].mean()

    best_region = region_yield.idxmax()

    print(
        f"\nBest performing region: {best_region}"
    )


if "Crop" in df.columns and "Yield" in df.columns:

    crop_yield = df.groupby("Crop")["Yield"].mean()

    best_crop = crop_yield.idxmax()

    print(
        f"Best performing crop: {best_crop}"
    )


# ------------------------------------------------------------
# 17. SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv(
    "agriculture_cleaned.csv",
    index=False
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print(
    "\nCleaned dataset saved as: agriculture_cleaned.csv"
)