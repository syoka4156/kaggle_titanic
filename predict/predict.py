import os

import numpy as np
import optuna
import pandas as pd
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    os.makedirs("output", exist_ok=True)

    train_df, test_df = import_data()
    test_passengerid_df = test_df[["PassengerId"]]

    train_df.dropna(subset=["Embarked"], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    train_df, test_df = extract_feature(train_df, test_df)
    train_df, test_df = standardization(train_df, test_df)

    kfold_splits = 5
    param_dict = optimize_param(train_df, kfold_splits)
    model_dict = cross_validation(train_df, param_dict, kfold_splits)

    test_df = pd.concat([test_passengerid_df, test_df], axis=1)
    predict_test_data(test_df, model_dict, kfold_splits)


def import_data():
    data_path = "../data"
    train_df = pd.read_csv(data_path + "/train.csv")
    test_df = pd.read_csv(data_path + "/test.csv")
    return train_df, test_df


def add_age_bin(df):
    copy_df = df.copy()
    max_age = int(copy_df["Age"].max())

    copy_df["Age_Bin"] = pd.cut(df["Age"], range(0, max_age + 1, 10)).astype(str)

    return copy_df


def extract_feature(train_df, test_df):
    length_train_df = len(train_df)

    train_test_df = pd.concat([train_df, test_df], axis=0)
    train_test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    train_test_df["Sex"] = train_test_df["Sex"].map({"male": 1, "female": 0})
    train_test_df = pd.get_dummies(train_test_df, prefix="Embarked", drop_first=True)

    survival_rate_dict = compose_dict_of_survival_rate_by_age(train_df)

    fill_nan_by_mean(train_test_df, ["Age", "Fare"])

    train_test_df = add_survival_rate_and_with_family(train_test_df, survival_rate_dict)

    return split_into_train_and_test(train_test_df, length_train_df)


def compose_dict_of_survival_rate_by_age(df):
    df = add_age_bin(df)
    rate_df = (
        df.dropna(subset=["Age_Bin"])
        .groupby("Age_Bin")["Survived"]
        .value_counts(normalize=True)
        .rename("Survival_Rate")
        .reset_index()
    )
    survival_rate_dict = (
        rate_df[rate_df["Survived"] == 1].set_index("Age_Bin").Survival_Rate.to_dict()
    )
    return survival_rate_dict


def fill_nan_by_mean(df, col_list):
    for col in col_list:
        df[col] = df[col].fillna(df[col].mean())


def add_survival_rate_and_with_family(df, survival_rate_dict):
    df = add_age_bin(df)
    df["Survival_Rate"] = df["Age_Bin"].map(survival_rate_dict)
    df.drop("Age_Bin", axis=1, inplace=True)

    df["with_Family"] = df.SibSp + df.Parch > 0
    df["with_Family"] = df["with_Family"].astype("int64")

    return df


def split_into_train_and_test(df, length_train_df):
    train_df = df.iloc[:length_train_df]
    test_df = df.iloc[length_train_df:]
    return train_df, test_df


def standardization(train_df, test_df):
    scaler = StandardScaler()
    scaler.fit(train_df.drop("Survived", axis=1))

    scaled_df_list = []

    for df in [train_df, test_df]:
        scaled_df = pd.DataFrame(
            scaler.transform(df.drop("Survived", axis=1)), columns=df.columns[1:]
        )
        scaled_df["Survived"] = df["Survived"]
        scaled_df_list.append(scaled_df)

    return scaled_df_list[0], scaled_df_list[1]


def optimize_param(df, kfold_splits):
    def objective_for_random_forest(trial):
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        clf = RandomForestClassifier(
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            **const_param_dict["random_forest"],
        )

        return cross_validation_for_optuna(clf)

    def objective_for_logistic_regression(trial):
        C = trial.suggest_int("C", 1, 100)
        solver_penalty_key = trial.suggest_categorical(
            "solver_penalty",
            (
                "newton-cg_l2",
                "lbfgs_l2",
                "liblinear_l1",
                "liblinear_l2",
                "sag_l2",
                "saga_l1",
                "saga_l2",
            ),
        )
        clf = LogisticRegression(
            C=C,
            **solver_penalty_param_dict[solver_penalty_key],
            **const_param_dict["logistic_regression"],
        )

        return cross_validation_for_optuna(clf)

    def cross_validation_for_optuna(clf):
        kf5 = KFold(n_splits=kfold_splits, shuffle=True, random_state=101)
        valid_accuracy_list = []
        test_accuracy_list = []

        for train_idx, valid_idx in kf5.split(X_df, y_df):
            X_train_sub = X_df.iloc[train_idx]
            X_valid_sub = X_df.iloc[valid_idx]
            y_train_sub = y_df.iloc[train_idx]
            y_valid_sub = y_df.iloc[valid_idx]

            clf.fit(X_train_sub, y_train_sub)

            predictions = clf.predict(X_valid_sub)
            valid_accuracy_list.append(accuracy_score(y_valid_sub, predictions))
            predictions = clf.predict(X_test)
            test_accuracy_list.append(accuracy_score(y_test, predictions))

        test_accuracy_mean_list.append(np.mean(test_accuracy_list))

        return np.mean(valid_accuracy_list)

    objective_dict = {
        "random_forest": objective_for_random_forest,
        "logistic_regression": objective_for_logistic_regression,
    }

    const_param_dict = {
        "random_forest": {"n_estimators": 800, "random_state": 101},
        "logistic_regression": {"max_iter": 700, "random_state": 101},
    }

    solver_penalty_param_dict = {
        "newton-cg_l2": {"solver": "newton-cg", "penalty": "l2"},
        "lbfgs_l2": {"solver": "lbfgs", "penalty": "l2"},
        "liblinear_l1": {"solver": "liblinear", "penalty": "l1"},
        "liblinear_l2": {"solver": "liblinear", "penalty": "l2"},
        "sag_l2": {"solver": "sag", "penalty": "l2"},
        "saga_l1": {"solver": "saga", "penalty": "l1"},
        "saga_l2": {"solver": "saga", "penalty": "l2"},
    }

    param_dict = {"random_sample": {}, "cluster_centroids": {}}
    optuna_result_dict = {
        "sample": [],
        "model": [],
        "param": [],
        "valid_accuracy": [],
        "test_accuracy": [],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("Survived", axis=1), df["Survived"], test_size=0.20, random_state=102
    )
    sample_dict = under_sample_dataset(X_train, y_train)

    for sample_name in ["random_sample", "cluster_centroids"]:
        X_df = sample_dict[sample_name][0]
        y_df = sample_dict[sample_name][1]

        for clf_name, objective_func in objective_dict.items():
            test_accuracy_mean_list = []
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.RandomSampler(seed=101)
            )
            study.optimize(objective_func, n_trials=100)

            if clf_name == "logistic_regression":
                param_dict[sample_name][clf_name] = {
                    **study.best_params,
                    **solver_penalty_param_dict[study.best_params["solver_penalty"]],
                    **const_param_dict[clf_name],
                }
                del param_dict[sample_name][clf_name]["solver_penalty"]
            else:
                param_dict[sample_name][clf_name] = {
                    **study.best_params,
                    **const_param_dict[clf_name],
                }

            optuna_result_dict["sample"].append(sample_name)
            optuna_result_dict["model"].append(clf_name)
            optuna_result_dict["param"].append(param_dict[sample_name][clf_name])
            optuna_result_dict["valid_accuracy"].append(study.best_value)
            optuna_result_dict["test_accuracy"].append(
                test_accuracy_mean_list[study.best_trial.number]
            )

    pd.DataFrame(optuna_result_dict, columns=optuna_result_dict.keys()).to_csv(
        "./output/optuna_result.csv"
    )

    return param_dict


def under_sample_dataset(X_train, y_train):
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=101)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)

    cc = ClusterCentroids(sampling_strategy=1.0, random_state=101)
    X_cc, y_cc = cc.fit_resample(X_train, y_train)

    return {"random_sample": [X_rus, y_rus], "cluster_centroids": [X_cc, y_cc]}


def cross_validation(df, param_dict, kfold_splits):
    model_dict = {
        "random_sample": {},
        "cluster_centroids": {},
    }
    result_dict = {
        "sample": [],
        "model": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    sample_dict = under_sample_dataset(df.drop("Survived", axis=1), df["Survived"])

    for sample_name, dataset_list in sample_dict.items():
        model_list = []

        X_df = dataset_list[0]
        y_df = dataset_list[1]
        kf5 = KFold(n_splits=kfold_splits, shuffle=True, random_state=101)

        clf_dict = {
            "random_forest": RandomForestClassifier(
                **param_dict[sample_name]["random_forest"]
            ),
            "logistic_regression": LogisticRegression(
                **param_dict[sample_name]["logistic_regression"]
            ),
        }

        for clf_name, clf in clf_dict.items():
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []

            for train_idx, valid_idx in kf5.split(X_df, y_df):
                X_train_sub = X_df.iloc[train_idx]
                X_valid_sub = X_df.iloc[valid_idx]
                y_train_sub = y_df.iloc[train_idx]
                y_valid_sub = y_df.iloc[valid_idx]

                clf.fit(X_train_sub, y_train_sub)

                predictions = clf.predict(X_valid_sub)
                accuracy_list.append(accuracy_score(y_valid_sub, predictions))
                precision_list.append(precision_score(y_valid_sub, predictions))
                recall_list.append(recall_score(y_valid_sub, predictions))
                f1_list.append(f1_score(y_valid_sub, predictions))

                model_list.append(clf)

            result_dict["sample"].append(sample_name)
            result_dict["model"].append(clf)
            result_dict["accuracy"].append(np.mean(accuracy_list))
            result_dict["precision"].append(np.mean(precision_list))
            result_dict["recall"].append(np.mean(recall_list))
            result_dict["f1"].append(np.mean(f1_list))

            model_dict[sample_name][clf_name] = model_list

    pd.DataFrame(result_dict, columns=result_dict.keys()).to_csv(
        "./output/titanic_result.csv"
    )

    return model_dict


def predict_test_data(test_df, model_dict, kfold_splits):
    for sample_name in ["random_sample", "cluster_centroids"]:
        for clf_name in ["random_forest", "logistic_regression"]:
            survived_arr = np.zeros(len(test_df))
            for i in range(kfold_splits):
                survived_arr += model_dict[sample_name][clf_name][i].predict(
                    test_df.drop(["Survived", "PassengerId"], axis=1)
                )

            survived_arr /= 5
            survived_arr = pd.Series((map(judge_survived, survived_arr)))
            pd.DataFrame(
                {"PassengerId": test_df["PassengerId"], "Survived": survived_arr}
            ).to_csv(f"./output/predictions_{sample_name}_{clf_name}.csv", index=False)


def judge_survived(x):
    if x > 0.5:
        return 1
    else:
        return 0


if __name__ == "__main__":
    main()
