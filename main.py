import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

# Importing and loading dataset into data variable.
import seaborn as sns
data = sns.load_dataset("penguins")

def main():
    # Indlæs og kig på data
    print(data.isna().sum())


if __name__ == "__main__":
    main()
