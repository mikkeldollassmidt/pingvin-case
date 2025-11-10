import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import seaborn as sns

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
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
    target = "species"
    data = data[features + [target]]
    print(data.head())


if __name__ == "__main__":
    main()
