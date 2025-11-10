import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():

    # Loading dataset.
    data = sns.load_dataset("penguins")

    #####################################
    # Initialize and look at the data   #
    #####################################

    # print(data.head())
    # print(data.info())
    # print(data.describe())
    # print(data.isna().sum())

    # Remove rows with missing values
    # print(data.dropna())

    # Specify the outcome
    # features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    # target = "species"
    # data = data[features + [target]]
    # print(data.head())

    # Hot Encode
    # data = pd.get_dummies(data, drop_first=True)
    # print(data.head())

    # Split Data
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    target = "species"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )

    print("Træningsdata:", X_train.shape)
    print("Testdata:", X_test.shape)

if __name__ == "__main__":
    main()
