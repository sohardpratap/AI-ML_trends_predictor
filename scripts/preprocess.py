# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("data/raw/shopping_trends.csv")  # Adjust path if needed

# Drop irrelevant columns
data = data.drop(columns=['Customer ID'])

# Encode categorical variables
categorical_columns = [
    'Gender', 'Category', 'Location', 'Size', 'Color', 'Season',
    'Subscription Status', 'Payment Method', 'Shipping Type',
    'Discount Applied', 'Promo Code Used', 'Preferred Payment Method'
]

# Map text values in 'Frequency of Purchases' to numerical categories
frequency_mapping = {
    'Weekly': 1,
    'Fortnightly': 2,
    'Every 3 Months': 3,
    'Every 6 Months': 6,
    'Annually': 12
}
data['Frequency of Purchases'] = data['Frequency of Purchases'].map(frequency_mapping)

# Normalize numerical columns
numerical_columns = ['Age', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Save processed data
data.to_csv("data/processed/processed_data.csv", index=False)
print("Preprocessing complete. Processed data saved to 'data/processed/processed_data.csv'.")
