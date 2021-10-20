# library doc string


# import libraries
import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve

from tests.test_churn_library import logging


def import_data(pth="data/bank_data.csv"):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """

    try:
        df = pd.read_csv(pth, index_col=0)
        logging.info("SUCCESS: Data imported")
    except FileNotFoundError as err:
        logging.error(err, "ERROR: FileNotFoundError")
        raise err

    return df


def prepare_data(df):
    """
       helper function to turn Attrition_Flag feature into binary feature named Churn

       input:
               df: pandas dataframe
       output:
               df: pandas dataframe with new column substituting Attrittion_Flag for Churn
    """
    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df = df.drop(columns=["Attrition_Flag"])
    logging.info("Converted Attrition_Flag feature into binary Churn feature")
    return df


def classify_columns(df):
    """
       helper function to label each of the dataframes columns as categorical or numerical based on the data types

       input:
               df: pandas dataframe
       output:
               all_cols: list of all the dataframe columns
               categorical_cols: list of all the dataframe columns that store categorical values
               numeric_cols: list of all the dataframe columns that store numerical values
    """

    all_cols = sorted(list(df.columns))
    categorical_cols = [col for col in all_cols if df[col].dtype == "O"]
    numeric_cols = list(set(all_cols) - set(categorical_cols))
    logging.info("Classified all dataframes columns into categorical or numerical")
    return all_cols, categorical_cols, numeric_cols


def perform_eda(df, categorical_cols, numeric_cols, pth=f"./images/eda/"):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    df = df[categorical_cols + numeric_cols]

    print("Dataframe shape = ", df.shape)
    print()
    print("Null counts per column:")
    print(df.isnull().sum())
    print()
    print("Dataframe describe")
    print(df.describe())

    # Univariate quant plots
    for col in numeric_cols:
        sns.displot(df[col])
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(pth + f"Numeric_Dist_{col}.png", dpi=400)
        plt.close()

    # Categorical column histograms
    for col in categorical_cols:
        sns.histplot(df[col])
        plt.xticks(rotation=45)
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(pth + f"Categorical_Dist_{col}.png", dpi=400)
        plt.close()

        sns.histplot(df[col], stat="percent")
        plt.xticks(rotation=45)
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(pth + f"Categorical_Dist_Normalized_{col}.png", dpi=400)
        plt.close()

    # Correlation plot
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    fig = plt.gcf()
    plt.figure(figsize=(20, 10))
    fig.savefig(pth + f"Correlation_Matrix.png", dpi=400)
    plt.close()

    # Pairplot
    sns.pairplot(df)
    fig = plt.gcf()
    plt.figure(figsize=(20, 10))
    fig.savefig(pth + "Pairplot.png", dpi=400)
    plt.close()

    logging.info(f"SUCCESS: Performed EDA in dataframe. Plots saved here {pth}")


def encoder_helper(df, category_lst, response=None):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or
            index y column]

    output:
            df: pandas dataframe with new columns for
    """
    encoder_cols = []
    for col in category_lst:
        temp_dict = df.groupby(col).mean()["Churn"].to_dict()
        new_col_name = col + "_Churn"
        if response:
            new_col_name = response
        df[new_col_name] = df[col].map(temp_dict)
        encoder_cols.append(new_col_name)

    logging.info(f"SUCCESS: Encoded categorical features into churn proportion columns")

    return df, encoder_cols


def cols_to_keep(numeric_cols: List, encoder_cols: List):
    """
    Combines numerical columns  with encoder columns and subtract not needed columns (Churn and CLIENTNUM)
    input:
            numeric_cols: list of dataframe numeric columns
            encoder_cols: list of encoded categorical columns

    output:
             keep_cols: list of columns to keep for training X array
    """
    keep_cols = list(set(numeric_cols + encoder_cols) - {"Churn"} - {"CLIENTNUM"})
    logging.info(f"Defined columns to keep for X train dataframe")

    return keep_cols


def perform_feature_engineering(df, keep_cols, response=None):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or
              index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y = df["Churn"]
    X = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info(f"SUCCESS: Split data into X_train, X_test, y_train, y_test")

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    print("random forest results")
    print("test results")
    print(classification_report(y_test, y_test_preds_rf))
    print("train results")
    print(classification_report(y_train, y_train_preds_rf))

    print("logistic regression results")
    print("test results")
    print(classification_report(y_test, y_test_preds_lr))
    print("train results")
    print(classification_report(y_train, y_train_preds_lr))

    logging.info(f"Printed classification report for both models")


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    fig = plt.gcf()
    plt.figure(figsize=(20, 10))

    save_pth = output_pth + "feature_importance.png"
    fig.savefig(save_pth, dpi=400)
    plt.close()

    logging.info(f"SUCCESS: Generated feature importance plot")


def roc_plot(rf, lrc, X_test, y_test, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rf, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    # Create plot title
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    fig = plt.gcf()

    save_pth = output_pth + "ROC_curve.png"
    fig.savefig(save_pth, dpi=400)
    plt.close()

    logging.info(f"SUCCESS: Generated ROC plot")


def train_models(X_train, X_test, y_train, y_test, results_pth="./images/results/", model_pth="./models/"):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info(f"SUCCESS: Fitted grid search random forest")

    lrc.fit(X_train, y_train)
    logging.info(f"SUCCESS: Fitted logistic regression")

    # Save best model for random forest
    cv_rfc_best = cv_rfc.best_estimator_

    y_train_preds_rf = cv_rfc_best.predict(X_train)
    y_test_preds_rf = cv_rfc_best.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    # Feature importance plot
    feature_importance_plot(cv_rfc_best, X_test, output_pth=results_pth)

    # ROC Plot
    roc_plot(cv_rfc_best, lrc, X_test, y_test, output_pth=results_pth)

    # save best model
    joblib.dump(cv_rfc_best, model_pth + "rfc_model.pkl")
    joblib.dump(lrc, model_pth + "logistic_model.pkl")
    logging.info(f"SUCCESS: Saved both models")


def main_pipeline():
    data = import_data()
    data = prepare_data(data)
    all_cols, categorical_cols, numeric_cols = classify_columns(data)
    perform_eda(data, categorical_cols, numeric_cols)
    data, encoder_cols = encoder_helper(data, category_lst=categorical_cols)
    keep_cols = cols_to_keep(numeric_cols, encoder_cols)
    X_train, X_test, y_train, y_test = perform_feature_engineering(data, keep_cols)
    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main_pipeline()
