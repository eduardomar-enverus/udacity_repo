import logging

import os
import pandas as pd
import tempfile

from churn_analysis.churn_library import perform_eda


# logging.basicConfig(
#     filename="logs/churn_library.log",
#     level=logging.INFO,
#     filemode="w",
#     format="%(name)s - %(levelname)s - %(message)s",
# )


def test_import(import_data):
    """
	test data import - this example is completed for you to assist with the other test functions
	"""
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_perform_eda():
    df = pd.DataFrame(
        {
            "CLIENTNUM": [
                768805383,
                818770008,
                713982108,
                769911858,
                709106358,
                713061558,
                810347208,
                818906208,
                710930508,
                719661558,
            ],
            "Customer_Age": [45, 49, 51, 40, 40, 44, 51, 32, 37, 48],
            "Gender": ["M", "F", "M", "F", "M", "M", "M", "M", "M", "M"],
            "Dependent_count": [3, 5, 3, 4, 3, 2, 4, 0, 3, 2],
            "Education_Level": [
                "High School",
                "Graduate",
                "Graduate",
                "High School",
                "Uneducated",
                "Graduate",
                "Unknown",
                "High School",
                "Uneducated",
                "Graduate",
            ],
            "Marital_Status": [
                "Married",
                "Single",
                "Married",
                "Unknown",
                "Married",
                "Married",
                "Married",
                "Unknown",
                "Single",
                "Single",
            ],
            "Income_Category": [
                "$60K - $80K",
                "Less than $40K",
                "$80K - $120K",
                "Less than $40K",
                "$60K - $80K",
                "$40K - $60K",
                "$120K +",
                "$60K - $80K",
                "$60K - $80K",
                "$80K - $120K",
            ],
            "Card_Category": ["Blue", "Blue", "Blue", "Blue", "Blue", "Blue", "Gold", "Silver", "Blue", "Blue"],
            "Months_on_book": [39, 44, 36, 34, 21, 36, 46, 27, 36, 36],
            "Total_Relationship_Count": [5, 6, 4, 3, 5, 3, 6, 2, 5, 6],
            "Months_Inactive_12_mon": [1, 1, 1, 4, 1, 1, 1, 2, 2, 3],
            "Contacts_Count_12_mon": [3, 2, 0, 1, 0, 2, 3, 2, 0, 3],
            "Credit_Limit": [12691.0, 8256.0, 3418.0, 3313.0, 4716.0, 4010.0, 34516.0, 29081.0, 22352.0, 11656.0],
            "Total_Revolving_Bal": [777, 864, 0, 2517, 0, 1247, 2264, 1396, 2517, 1677],
            "Avg_Open_To_Buy": [11914.0, 7392.0, 3418.0, 796.0, 4716.0, 2763.0, 32252.0, 27685.0, 19835.0, 9979.0],
            "Total_Amt_Chng_Q4_Q1": [1.335, 1.541, 2.594, 1.405, 2.175, 1.376, 1.975, 2.204, 3.355, 1.524],
            "Total_Trans_Amt": [1144, 1291, 1887, 1171, 816, 1088, 1330, 1538, 1350, 1441],
            "Total_Trans_Ct": [42, 33, 20, 20, 28, 24, 31, 36, 24, 32],
            "Total_Ct_Chng_Q4_Q1": [1.625, 3.714, 2.333, 2.333, 2.5, 0.846, 0.722, 0.7140000000000001, 1.182, 0.882],
            "Avg_Utilization_Ratio": [0.061, 0.105, 0.0, 0.76, 0.0, 0.311, 0.066, 0.048, 0.113, 0.144],
            "Churn": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    categorical_cols = ["Education_Level", "Marital_Status"]
    numeric_cols = ["Dependent_count", "Total_Revolving_Bal"]

    temp_dir = tempfile.TemporaryDirectory()
    pth = temp_dir.name + "/"
    perform_eda(df, categorical_cols, numeric_cols, pth)

    filenames = [f for f in os.listdir(pth) if os.path.isfile(os.path.join(pth, f))]
    filenames = sorted(filenames)
    temp_dir.cleanup()

    expected_filenames = [
        "Categorical_Dist_Education_Level.png",
        "Categorical_Dist_Marital_Status.png",
        "Categorical_Dist_Normalized_Education_Level.png",
        "Categorical_Dist_Normalized_Marital_Status.png",
        "Correlation_Matrix.png",
        "Numeric_Dist_Dependent_count.png",
        "Numeric_Dist_Total_Revolving_Bal.png",
        "Pairplot.png",
    ]

    assert filenames == expected_filenames


def test_encoder_helper():

    pass


def test_perform_feature_engineering(perform_feature_engineering):
    """
	test perform_feature_engineering
	"""


def test_train_models(train_models):
    """
	test train_models
	"""


if __name__ == "__main__":
    pass
