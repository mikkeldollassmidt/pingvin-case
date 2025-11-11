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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    # features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    # target = "species"

    # X = data[features]
    # y = data[target]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=69
    # )

    # print("Træningsdata:", X_train.shape)
    # print("Testdata:", X_test.shape)

    # Train the machine
    data = data.dropna()
    
    le_sex = LabelEncoder()
    data["sex"] = le_sex.fit_transform(data["sex"])

    le_species = LabelEncoder()
    data["species"] = le_species.fit_transform(data["species"])
    
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    target = "species"


    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    model_tree.fit(X_train, y_train)


    # Evaluating data
    y_pred_tree = model_tree.predict(X_test)
    print("1. DT Accuracy:", round(accuracy_score(y_test, y_pred_tree), 2))
    print("2. DT Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
    print("3. DT Classification Report:\n", classification_report(y_test, y_pred_tree)) 
       
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    print("1. KNN Accuracy:", round(accuracy_score(y_test, y_pred_knn), 2))
    print("2. KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
    print("3. KNN Classification Report:\n", classification_report(y_test, y_pred_knn))


if __name__ == "__main__":
    main()
