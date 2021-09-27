# library doc string


# import libraries
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth, index_col=0)
        logging.info("SUCCESS: Data imported")

    except FileNotFoundError as err:
        logging.error(err, "ERROR: FileNotFoundError")
        raise err

    return df


data = import_data('data/bank_data.csv')

def prepare_data(df):
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df = df.drop(columns = ["Attrition_Flag"])
    return df

data = prepare_data(data)


def classify_columns(df):
    all_cols = sorted(list(df.columns))
    categorical_cols = [col for col in all_cols if df[col].dtype == "O"]
    numeric_cols = list(set(all_cols) - set(categorical_cols))
    return all_cols, categorical_cols, numeric_cols

all_cols, categorical_cols, numeric_cols = classify_columns(data)

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    all_cols, categorical_cols, numeric_cols = classify_columns(df)

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
        fig.savefig(f'./images/eda/Numeric_Dist_{col}.png', dpi=400)
        plt.clf()

    # Categorical column histograms
    for col in categorical_cols:
        sns.histplot(df[col])
        plt.xticks(rotation=45)
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(f'./images/eda/Categorical_Dist_{col}.png', dpi=400)
        plt.clf()

        sns.histplot(df[col], stat="percent")
        plt.xticks(rotation=45)
        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        fig.savefig(f'./images/eda/Categorical_Dist_Normalized_{col}.png', dpi=400)
        plt.clf()

    # Correlation plot
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig = plt.gcf()
    plt.figure(figsize=(20, 10))
    fig.savefig(f'./images/eda/Correlation_Matrix.png', dpi=400)
    plt.clf()

    # Pairplot
    sns.pairplot(df)
    fig = plt.gcf()
    plt.figure(figsize=(20, 10))
    fig.savefig(f'./images/eda/Pairplot.png', dpi=400)
    plt.clf()


# perform_eda(data)


def encoder_helper(df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for col in category_lst:
        temp_dict = df.groupby(col).mean()['Churn'].to_dict()
        new_col_name = col+'_Churn'
        if response:
            new_col_name = response
        df[new_col_name] = df[col].map(temp_dict)

    return df

data = encoder_helper(data, category_lst=categorical_cols)
KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = df[KEEP_COLS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = perform_feature_engineering(data)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
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
    '''
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))



def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    pass