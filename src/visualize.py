import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    os.makedirs("../output", exist_ok=True)

    train_df, test_df = import_data()

    train_df.dropna(subset=["Embarked"], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    explore_data(train_df, test_df)


def import_data():
    data_path = "./data/titanic"
    train_df = pd.read_csv(data_path + "/train.csv")
    test_df = pd.read_csv(data_path + "/test.csv")
    return train_df, test_df


def explore_data(train_df, test_df):
    vs_survived_list = ["Age", "Fare"]
    num_rate_list = ["Pclass", "Sex", "Embarked"]
    check_null([train_df, test_df])
    visualize_num_of_survivors(train_df)

    visualize_col_vs_survived(
        train_df[vs_survived_list + ["Survived"]], vs_survived_list
    )

    visualize_num_and_rate_of_survivors(
        train_df[num_rate_list + ["Survived"]], num_rate_list
    )

    visualize_scatter_whether_survived(train_df, "Age", "Fare")

    age_bin_added_df = add_age_bin(train_df)
    visualize_survival_rate(age_bin_added_df, "Age_Bin", True)


def check_null(df_list):
    for df in df_list:
        print(df.isnull().sum())


def visualize_num_of_survivors(df):
    print(
        df["Survived"].value_counts().rename(index={1: "Survived", 0: "Not Survived"})
    )
    sns.catplot(x="Survived", data=df, kind="count", aspect=1.8, height=5.1)
    plt.suptitle("Number of Survivors")
    plt.savefig("../output/num_of_survivors.png")
    plt.close()


def visualize_col_vs_survived(df, col_list):
    for col in col_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
        sns.histplot(data=df, x=col, hue="Survived", ax=ax1)
        sns.boxplot(data=df, x="Survived", y=col, ax=ax2)
        plt.suptitle(f"{col} vs Survived")
        plt.savefig(f"../output/{col}_vs_Survived.png")
        plt.close()


def visualize_num_and_rate_of_survivors(df, col_list):
    for col in col_list:
        sns.catplot(x=col, data=df, kind="count", aspect=1.8, height=5.1)
        plt.ylabel("Number of Passengers")
        plt.suptitle(f"Number of Survivors in {col}")
        plt.savefig(f"../output/num_of_survivors_in_{col}.png")
        plt.close()

        visualize_survival_rate(df, col)


def visualize_survival_rate(df, col, sort_bool=False):
    df_rate = (
        df.groupby(col, sort=sort_bool)["Survived"]
        .value_counts(normalize=True)
        .rename("Survival_Rate")
        .reset_index()
    )
    sns.catplot(
        x=col,
        y="Survival_Rate",
        data=df_rate[df_rate["Survived"] == 1],
        kind="bar",
        aspect=1.8,
        height=5.1,
    )
    plt.suptitle(f"Survival Rate in {col}")
    plt.savefig(f"../output/survival_rate_in_{col}.png")
    plt.close()


def visualize_scatter_whether_survived(df, x_axis, y_axis):
    plt.figure(figsize=(11, 6))
    sns.scatterplot(x=x_axis, y=y_axis, hue="Survived", data=df)
    plt.suptitle(f"{x_axis} vs {y_axis} in Survived")
    plt.savefig(f"../output/{x_axis}_vs_{y_axis}_in_survived")
    plt.close()


def add_age_bin(df):
    copy_df = df.copy()
    max_age = int(copy_df["Age"].max())

    copy_df["Age_Bin"] = pd.cut(df["Age"], range(0, max_age + 1, 10)).astype(str)

    return copy_df


if __name__ == "__main__":
    main()
