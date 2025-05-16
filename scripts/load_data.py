# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'shopping_trends.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv("data/raw/shopping_trends.csv")

# Display basic information
print("Data Overview:")
print(data.head())  # View the first 5 rows
print("\nDataset Info:")
print(data.info())  # Check data types and non-null values
print("\nSummary Statistics:")
print(data.describe(include='all'))  # Summary for numerical and categorical data

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Visualize distributions of key features
numeric_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']
categorical_columns = ['Gender', 'Category', 'Location', 'Season', 'Subscription Status', 'Payment Method']

# Plot numeric distributions
for column in numeric_columns:
    if column in data.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.show()

# Plot categorical distributions
for column in categorical_columns:
    if column in data.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=data[column], order=data[column].value_counts().index)
        plt.title(f'Distribution of {column}')
        plt.show()
