# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Project to predict customer churn. Performs EDA and trains a random forest and logistic regression classifiers to
classify churn on provided `bank_data.csv`

## Running Files
Please run `conda env create -f environment.yml` to install conda environment and all dependencies. After that you can
simply run `python -m churn_analysis.churn_library` to run application.


## Unit Testing
I decided to use pytest to carry out the unit testing within this project because it is what we use at my current 
company so wanted to use similar interface. To run unit testing simply type `pytest tests` and pytest will generate
an interface and notify you of which tests are passing or not.